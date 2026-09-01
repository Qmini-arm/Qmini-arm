/*
 * uHand USB-serial bridge firmware
 *
 * Direct computer control through the UNO Type-B USB cable. The hardware
 * Serial is used for binary packets, so this sketch intentionally emits no
 * debug text on Serial.
 *
 * Packet: AA 77 | function | data_length | data | checksum
 * checksum = bitwise NOT of (function + data_length + data), low byte
 *
 * 0x01: five finger angles plus one ignored compatibility byte
 * 0x02: buzzer frequency/time, four little-endian bytes
 * 0x03: RGB, three bytes
 * 0x11: read five finger targets plus one compatibility byte
 */

#include <Arduino.h>
#include <FastLED.h>
#include <Servo.h>

const uint8_t HEADER_1 = 0xAA;
const uint8_t HEADER_2 = 0x77;
const uint8_t FUNC_SET_SERVO = 0x01;
const uint8_t FUNC_SET_BUZZER = 0x02;
const uint8_t FUNC_SET_RGB = 0x03;
const uint8_t FUNC_READ_ANGLE = 0x11;
const uint8_t MAX_DATA = 20;
const uint8_t FINGER_COUNT = 5;
const uint8_t LEGACY_SERVO_DATA_LENGTH = 6;
const uint8_t RESERVED_PROTOCOL_VALUE = 90;

// The PC program already smooths the target and limits each command step.
// A 10% firmware filter made the hand feel sluggish because it added a
// second long delay.  This response is intentionally faster while still
// avoiding an instantaneous 0..180 degree jump in the command stream.
const uint32_t SERVO_UPDATE_INTERVAL_MS = 20;
const float SERVO_RESPONSE = 0.30f;

// H1..H5 only. D2 is intentionally not attached or controlled.
const uint8_t servo_pins[FINGER_COUNT] = {7, 6, 5, 4, 3};
const uint8_t buzzer_pin = 11;
const uint8_t rgb_pin = 13;

Servo servos[FINGER_COUNT];
CRGB rgb_leds[1];

// Logical angles for H1..H5 only.
uint8_t target_angles[FINGER_COUNT] = {180, 180, 180, 180, 180};
float actual_angles[FINGER_COUNT] = {180, 180, 180, 180, 180};
const uint8_t angle_min[FINGER_COUNT] = {0, 0, 0, 0, 0};
const uint8_t angle_max[FINGER_COUNT] = {180, 180, 180, 180, 180};

enum ParserState {
  WAIT_HEADER_1,
  WAIT_HEADER_2,
  READ_FUNCTION,
  READ_LENGTH,
  READ_DATA,
  READ_CHECKSUM,
};

ParserState parser_state = WAIT_HEADER_1;
uint8_t packet_function = 0;
uint8_t packet_length = 0;
uint8_t packet_index = 0;
uint8_t packet_data[MAX_DATA] = {0};

uint8_t packet_checksum(const uint8_t function, const uint8_t length, const uint8_t *data) {
  uint8_t sum = function + length;
  for (uint8_t i = 0; i < length; ++i) {
    sum = sum + data[i];
  }
  return static_cast<uint8_t>(~sum);
}

void reset_parser() {
  parser_state = WAIT_HEADER_1;
  packet_function = 0;
  packet_length = 0;
  packet_index = 0;
}

void send_packet(const uint8_t function, const uint8_t length, const uint8_t *data) {
  Serial.write(HEADER_1);
  Serial.write(HEADER_2);
  Serial.write(function);
  Serial.write(length);
  for (uint8_t i = 0; i < length; ++i) {
    Serial.write(data[i]);
  }
  Serial.write(packet_checksum(function, length, data));
}

void handle_packet() {
  if (packet_function == FUNC_SET_SERVO &&
      packet_length == LEGACY_SERVO_DATA_LENGTH) {
    for (uint8_t i = 0; i < FINGER_COUNT; ++i) {
      target_angles[i] = constrain(packet_data[i], angle_min[i], angle_max[i]);
    }
    // packet_data[5] is required by the legacy frame but intentionally ignored.
    return;
  }

  if (packet_function == FUNC_SET_BUZZER && packet_length == 4) {
    uint16_t frequency = static_cast<uint16_t>(packet_data[0]) |
                         (static_cast<uint16_t>(packet_data[1]) << 8);
    uint16_t duration = static_cast<uint16_t>(packet_data[2]) |
                        (static_cast<uint16_t>(packet_data[3]) << 8);
    if (frequency == 0) {
      noTone(buzzer_pin);
    } else {
      tone(buzzer_pin, frequency, duration);
    }
    return;
  }

  if (packet_function == FUNC_SET_RGB && packet_length == 3) {
    rgb_leds[0] = CRGB(packet_data[0], packet_data[1], packet_data[2]);
    FastLED.show();
    return;
  }

  if (packet_function == FUNC_READ_ANGLE && packet_length == 0) {
    uint8_t response[LEGACY_SERVO_DATA_LENGTH];
    for (uint8_t i = 0; i < FINGER_COUNT; ++i) {
      response[i] = target_angles[i];
    }
    response[5] = RESERVED_PROTOCOL_VALUE;
    send_packet(FUNC_READ_ANGLE, LEGACY_SERVO_DATA_LENGTH, response);
  }
}

void feed_serial_byte(const uint8_t value) {
  switch (parser_state) {
    case WAIT_HEADER_1:
      if (value == HEADER_1) {
        parser_state = WAIT_HEADER_2;
      }
      break;

    case WAIT_HEADER_2:
      if (value == HEADER_2) {
        parser_state = READ_FUNCTION;
      } else if (value != HEADER_1) {
        parser_state = WAIT_HEADER_1;
      }
      break;

    case READ_FUNCTION:
      if (value == FUNC_SET_SERVO || value == FUNC_SET_BUZZER ||
          value == FUNC_SET_RGB || value == FUNC_READ_ANGLE) {
        packet_function = value;
        parser_state = READ_LENGTH;
      } else {
        reset_parser();
      }
      break;

    case READ_LENGTH:
      if (value > MAX_DATA) {
        reset_parser();
      } else {
        packet_length = value;
        packet_index = 0;
        parser_state = packet_length == 0 ? READ_CHECKSUM : READ_DATA;
      }
      break;

    case READ_DATA:
      packet_data[packet_index++] = value;
      if (packet_index >= packet_length) {
        parser_state = READ_CHECKSUM;
      }
      break;

    case READ_CHECKSUM:
      if (value == packet_checksum(packet_function, packet_length, packet_data)) {
        handle_packet();
      }
      reset_parser();
      break;
  }
}

void receive_serial() {
  while (Serial.available() > 0) {
    feed_serial_byte(static_cast<uint8_t>(Serial.read()));
  }
}

void servo_control() {
  static uint32_t last_tick = 0;
  const uint32_t now = millis();
  if (now - last_tick < SERVO_UPDATE_INTERVAL_MS) {
    return;
  }
  last_tick = now;

  for (uint8_t i = 0; i < FINGER_COUNT; ++i) {
    actual_angles[i] = actual_angles[i] * (1.0f - SERVO_RESPONSE) +
                      target_angles[i] * SERVO_RESPONSE;
    if (abs(actual_angles[i] - target_angles[i]) < 0.5f) {
      actual_angles[i] = target_angles[i];
    }
    actual_angles[i] = constrain(actual_angles[i], angle_min[i], angle_max[i]);

    // Same thumb direction as the course PC serial firmware.
    const float physical_angle = i == 0 ? 180.0f - actual_angles[i] : actual_angles[i];
    servos[i].write(static_cast<int>(physical_angle));
  }
}

void setup() {
  Serial.begin(115200);

  for (uint8_t i = 0; i < FINGER_COUNT; ++i) {
    servos[i].attach(servo_pins[i], 500, 2500);
    const float physical_angle = i == 0 ? 180.0f - actual_angles[i] : actual_angles[i];
    servos[i].write(static_cast<int>(physical_angle));
  }

  pinMode(buzzer_pin, OUTPUT);
  FastLED.addLeds<WS2812, rgb_pin, GRB>(rgb_leds, 1);
  rgb_leds[0] = CRGB::White;
  FastLED.show();
}

void loop() {
  receive_serial();
  servo_control();
  delay(1);
}
