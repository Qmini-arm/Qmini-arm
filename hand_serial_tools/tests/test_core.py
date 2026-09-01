from __future__ import annotations

import inspect
import threading
import time
import unittest

from uhand import (
    DEFAULT_BAUD,
    GESTURES,
    FingerCalibration,
    FingerValueError,
    UHand,
    build_finger_packet,
    connect,
    validate_finger_angles,
)


class FakeSerial:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> int:
        self.writes.append(bytes(data))
        return len(data)

    def close(self) -> None:
        self.closed = True


class BlockingFirstSerial(FakeSerial):
    def __init__(self) -> None:
        super().__init__()
        self.first_write_started = threading.Event()
        self.release_first_write = threading.Event()

    def write(self, data: bytes) -> int:
        if not self.writes:
            self.first_write_started.set()
            if not self.release_first_write.wait(timeout=1.0):
                raise TimeoutError("test did not release first write")
        return super().write(data)


class PacketTests(unittest.TestCase):
    def test_open_packet_has_fixed_reserved_byte_and_checksum(self) -> None:
        packet = build_finger_packet([180, 180, 180, 180, 180])
        self.assertEqual(packet, bytes.fromhex("AA 77 01 06 B4 B4 B4 B4 B4 5A 1A"))
        self.assertEqual(packet[9], 90)

    def test_six_values_are_rejected(self) -> None:
        with self.assertRaises(FingerValueError):
            build_finger_packet([1, 2, 3, 4, 5, 6])

    def test_angles_are_not_silently_clamped(self) -> None:
        with self.assertRaises(FingerValueError):
            validate_finger_angles([181, 0, 0, 0, 0])


class CalibrationTests(unittest.TestCase):
    def test_closure_mapping_uses_per_finger_endpoints(self) -> None:
        calibration = FingerCalibration(
            open_angles=(170, 160, 150, 140, 130),
            closed_angles=(10, 20, 30, 40, 50),
        )
        self.assertEqual(
            calibration.angles_for_closures((0.0, 0.25, 0.5, 0.75, 1.0)),
            (170, 125, 90, 65, 50),
        )

    def test_presets_have_exactly_five_normalized_values(self) -> None:
        for closures in GESTURES.values():
            self.assertEqual(len(closures), 5)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in closures))


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.serial = FakeSerial()
        self.hand = UHand(
            self.serial,
            port="fake",
        )

    def tearDown(self) -> None:
        self.hand.close()

    def test_set_fingers_writes_one_packet(self) -> None:
        target = self.hand.set_fingers([10, 20, 30, 40, 50])
        self.assertEqual(target, (10, 20, 30, 40, 50))
        self.assertEqual(self.serial.writes[-1], build_finger_packet(target))

    def test_set_one_finger_preserves_others(self) -> None:
        self.hand.set_fingers([10, 20, 30, 40, 50])
        target = self.hand.set_finger("H2", 99)
        self.assertEqual(target, (10, 99, 30, 40, 50))

    def test_gesture_uses_calibrated_five_finger_target(self) -> None:
        target = self.hand.gesture("victory", duration=0)
        self.assertEqual(target, (0, 180, 180, 0, 0))
        self.assertEqual(len(target), 5)

    def test_close_does_not_send_an_extra_pose(self) -> None:
        self.hand.set_fingers([20, 20, 20, 20, 20])
        writes_before = len(self.serial.writes)
        self.hand.close()
        self.assertEqual(len(self.serial.writes), writes_before)
        self.assertTrue(self.serial.closed)

    def test_realtime_queue_drops_stale_unsent_target(self) -> None:
        self.hand.close()
        serial = BlockingFirstSerial()
        self.hand = UHand(
            serial,
            port="fake",
        )
        first = (10, 10, 10, 10, 10)
        stale = (20, 20, 20, 20, 20)
        newest = (30, 30, 30, 30, 30)

        self.hand.command_fingers(first)
        self.assertTrue(serial.first_write_started.wait(timeout=0.5))
        self.hand.command_fingers(stale)
        self.hand.command_fingers(newest)
        serial.release_first_write.set()

        deadline = time.monotonic() + 0.5
        while len(serial.writes) < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertEqual(
            serial.writes,
            [build_finger_packet(first), build_finger_packet(newest)],
        )


class ConnectApiTests(unittest.TestCase):
    def test_connect_is_direct_usb_only(self) -> None:
        parameters = inspect.signature(connect).parameters
        self.assertNotIn("transport", parameters)
        self.assertNotIn("baud", parameters)
        self.assertEqual(DEFAULT_BAUD, 115200)


if __name__ == "__main__":
    unittest.main()
