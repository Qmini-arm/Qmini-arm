"""Basic safe motion API for the six-axis CDS55xx arm.

No demonstrations, predefined pose sequences, interactive calibration, or CLI
code lives in this module.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator


HEADER = b"\xFF\xFF"
BROADCAST_ID = 0xFE
INST_READ = 0x02
INST_SYNC_WRITE = 0x83

ANGLE_LIMIT_ADDR = 0x06
MAX_TORQUE_ADDR = 0x0E
TORQUE_ENABLE_ADDR = 0x18
CONTROL_GAINS_ADDR = 0x1A
GOAL_POSITION_ADDR = 0x1E
RUNTIME_STATUS_ADDR = 0x20
PRESENT_POSITION_ADDR = 0x24
MINIMUM_PWM_ADDR = 0x30

POSITION_MAX = 1023
DEFAULT_BAUD = 115_200
SERVO_IDS = (1, 2, 3, 4, 5, 6)

# Calibrated feedback center and independent joint safety limits.
CENTER = {1: 812, 2: 122, 3: 144, 4: 481, 5: 359, 6: 88}
SAFE_LIMITS = {
    1: (640, 1000),
    2: (30, 350),
    3: (100, 600),
    4: (330, 600),
    5: (250, 470),
    6: (0, 200),
}

ERROR_BITS = {
    6: "指令错误",
    5: "过载",
    4: "校验和错误",
    3: "指令超范围",
    2: "过热",
    1: "角度超范围",
    0: "过压或欠压",
}


class SafetyError(RuntimeError):
    """A configured or observed arm state violated a safety constraint."""


@dataclass(frozen=True)
class StatusPacket:
    servo_id: int
    error: int
    params: bytes
    frame: bytes


def decode_error(error: int) -> str:
    labels = [label for bit, label in ERROR_BITS.items() if error & (1 << bit)]
    return ",".join(labels) if labels else "OK"


def build_packet(
    servo_id: int,
    instruction: int,
    params: Iterable[int] = (),
) -> bytes:
    if not 0 <= servo_id <= BROADCAST_ID:
        raise ValueError("servo_id必须在0..254")
    values = [int(value) & 0xFF for value in params]
    length = len(values) + 2
    checksum = (~(servo_id + length + instruction + sum(values))) & 0xFF
    return HEADER + bytes([servo_id, length, instruction, *values, checksum])


def checksum_is_valid(frame: bytes) -> bool:
    return len(frame) >= 6 and (sum(frame[2:]) & 0xFF) == 0xFF


def _extract_status_packets(
    data: bytes,
    *,
    expected_id: int,
    expected_param_count: int,
    request: bytes,
) -> list[StatusPacket]:
    packets: list[StatusPacket] = []
    offset = 0
    while offset + 4 <= len(data):
        header_at = data.find(HEADER, offset)
        if header_at < 0 or header_at + 4 > len(data):
            break
        length = data[header_at + 3]
        if length < 2 or length > 64:
            offset = header_at + 1
            continue
        frame_end = header_at + 4 + length
        if frame_end > len(data):
            break
        frame = data[header_at:frame_end]
        offset = frame_end
        if frame == request or frame[2] != expected_id or not checksum_is_valid(frame):
            continue
        params = frame[5:-1]
        if len(params) == expected_param_count:
            packets.append(StatusPacket(frame[2], frame[4], params, frame))
    return packets


def _exchange(
    serial_port: object,
    request: bytes,
    *,
    expected_id: int,
    expected_param_count: int,
    timeout: float,
) -> tuple[StatusPacket | None, bytes]:
    serial_port.reset_input_buffer()
    serial_port.write(request)
    serial_port.flush()
    deadline = time.monotonic() + timeout
    received = bytearray()
    while time.monotonic() < deadline:
        waiting = int(getattr(serial_port, "in_waiting", 0))
        chunk = serial_port.read(waiting if waiting > 0 else 1)
        if not chunk:
            continue
        received.extend(chunk)
        packets = _extract_status_packets(
            bytes(received),
            expected_id=expected_id,
            expected_param_count=expected_param_count,
            request=request,
        )
        if packets:
            return packets[-1], bytes(received)
    return None, bytes(received)


def _discover_ports() -> list[object]:
    from serial.tools import list_ports

    return list(list_ports.comports())


def _choose_port(requested: str) -> str:
    if requested.lower() != "auto":
        return requested
    tokens = (
        "usbserial",
        "ttyusb",
        "ttyacm",
        "wchusbserial",
        "ch340",
        "ch341",
        "cp210",
        "ftdi",
    )
    candidates: list[str] = []
    for port in _discover_ports():
        device = str(getattr(port, "device", ""))
        combined = " ".join(
            str(getattr(port, key, "")) for key in ("device", "description", "hwid")
        ).lower()
        supported = device.startswith(
            ("/dev/cu.", "/dev/ttyUSB", "/dev/ttyACM", "/dev/serial/")
        )
        if supported and any(token in combined for token in tokens):
            candidates.append(device)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError("无法自动识别UP-Debugger，请显式传入port")
    raise RuntimeError(f"发现多个候选串口{candidates}，请显式传入port")


def validate_configuration() -> None:
    if set(CENTER) != set(SERVO_IDS) or set(SAFE_LIMITS) != set(SERVO_IDS):
        raise SafetyError("中心、限位和舵机ID集合不一致")
    validate_positions(CENTER, "正式中心")


def validate_positions(values: dict[int, int], context: str = "位置") -> None:
    if set(values) != set(SERVO_IDS):
        raise SafetyError(f"{context}必须完整包含ID 1..6")
    violations = []
    for servo_id in SERVO_IDS:
        lower, upper = SAFE_LIMITS[servo_id]
        if not lower <= values[servo_id] <= upper:
            violations.append(
                f"ID{servo_id}={values[servo_id]}不在[{lower},{upper}]"
            )
    if violations:
        raise SafetyError(f"{context}越界：" + "；".join(violations))


class CDSArm:
    """Safe six-axis operations around an already-open pyserial port."""

    def __init__(
        self,
        serial_port: object,
        timeout: float = 0.08,
        port_name: str | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout必须大于0")
        self.serial = serial_port
        self.timeout = timeout
        self.port_name = port_name or str(getattr(serial_port, "port", ""))

    def close(self) -> None:
        self.serial.close()

    def __enter__(self) -> CDSArm:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def read_registers(
        self,
        servo_id: int,
        address: int,
        size: int,
        retries: int = 2,
    ) -> bytes:
        if retries < 0 or size < 1:
            raise ValueError("retries不能为负且size必须大于0")
        request = build_packet(servo_id, INST_READ, [address, size])
        received_attempts: list[bytes] = []
        for attempt in range(retries + 1):
            status, raw_rx = _exchange(
                self.serial,
                request,
                expected_id=servo_id,
                expected_param_count=size,
                timeout=self.timeout,
            )
            received_attempts.append(raw_rx)
            if status is not None:
                if status.error:
                    raise RuntimeError(
                        f"ID{servo_id}报告0x{status.error:02X} "
                        f"({decode_error(status.error)})"
                    )
                return status.params
            if attempt < retries:
                time.sleep(0.004)
        details = [raw.hex(" ").upper() if raw else "无数据" for raw in received_attempts]
        raise RuntimeError(
            f"ID{servo_id}连续{retries + 1}次读取0x{address:02X}无有效应答；"
            f"RX={details}"
        )

    def read_u16(self, servo_id: int, address: int) -> int:
        data = self.read_registers(servo_id, address, 2)
        return data[0] | (data[1] << 8)

    def read_positions(self) -> dict[int, int]:
        positions = {
            servo_id: self.read_u16(servo_id, PRESENT_POSITION_ADDR)
            for servo_id in SERVO_IDS
        }
        validate_positions(positions, "实际反馈")
        return positions

    def sample_positions(self, rounds: int = 3) -> dict[int, list[int]]:
        if rounds < 1:
            raise ValueError("rounds必须大于0")
        samples = {servo_id: [] for servo_id in SERVO_IDS}
        for _ in range(rounds):
            actual = self.read_positions()
            for servo_id in SERVO_IDS:
                samples[servo_id].append(actual[servo_id])
            time.sleep(0.04)
        return samples

    def stable_sample(
        self,
        *,
        rounds: int = 3,
        max_drift: int = 12,
        allow_zero: bool = False,
    ) -> dict[int, int]:
        if max_drift < 0:
            raise ValueError("max_drift不能为负")
        samples = self.sample_positions(rounds)
        result = {}
        for servo_id, values in samples.items():
            if max(values) - min(values) > max_drift:
                raise SafetyError(f"ID{servo_id}采样漂移过大：{values}")
            if not allow_zero and all(value == 0 for value in values):
                raise SafetyError(f"ID{servo_id}连续读到0，无法排除反馈盲区")
            result[servo_id] = values[-1]
        return result

    @staticmethod
    def _sync_frame(
        address: int,
        data_size: int,
        values: dict[int, list[int]],
    ) -> bytes:
        params = [address, data_size]
        for servo_id in sorted(values):
            data = values[servo_id]
            if len(data) != data_size:
                raise ValueError("同步写数据长度错误")
            params.extend([servo_id, *data])
        return build_packet(BROADCAST_ID, INST_SYNC_WRITE, params)

    def _write_frame(self, frame: bytes) -> None:
        self.serial.reset_input_buffer()
        self.serial.write(frame)
        self.serial.flush()

    def send_goals(
        self,
        positions: dict[int, int],
        speed: int,
        *,
        verify: bool = True,
    ) -> bytes:
        if not 1 <= speed <= POSITION_MAX:
            raise ValueError("speed必须在1..1023；0等于全速")
        validate_positions(positions, "待发送目标")
        values = {
            servo_id: [
                raw & 0xFF,
                (raw >> 8) & 0xFF,
                speed & 0xFF,
                (speed >> 8) & 0xFF,
            ]
            for servo_id, raw in positions.items()
        }
        frame = self._sync_frame(GOAL_POSITION_ADDR, 4, values)
        self._write_frame(frame)
        if verify:
            time.sleep(0.02)
            readback = {
                servo_id: self.read_u16(servo_id, GOAL_POSITION_ADDR)
                for servo_id in SERVO_IDS
            }
            mismatches = {
                servo_id: (positions[servo_id], readback[servo_id])
                for servo_id in SERVO_IDS
                if positions[servo_id] != readback[servo_id]
            }
            if mismatches:
                raise SafetyError(f"目标寄存器回读不一致：{mismatches}")
        return frame

    def torque_states(self) -> dict[int, int]:
        return {
            servo_id: self.read_registers(servo_id, TORQUE_ENABLE_ADDR, 1)[0]
            for servo_id in SERVO_IDS
        }

    def verify_position_modes(self) -> dict[int, tuple[int, int]]:
        configured = {}
        for servo_id in SERVO_IDS:
            data = self.read_registers(servo_id, ANGLE_LIMIT_ADDR, 4)
            cw = data[0] | (data[1] << 8)
            ccw = data[2] | (data[3] << 8)
            if cw == 0 and ccw == 0:
                raise SafetyError(f"ID{servo_id}处于连续旋转模式")
            lower, upper = SAFE_LIMITS[servo_id]
            if not 0 <= cw < ccw <= POSITION_MAX or cw > lower or ccw < upper:
                raise SafetyError(
                    f"ID{servo_id}内部限位[{cw},{ccw}]不能覆盖"
                    f"安全范围[{lower},{upper}]"
                )
            configured[servo_id] = (cw, ccw)
        return configured

    def establish_hold(
        self,
        current: dict[int, int],
        speed: int,
        torque_states: dict[int, int],
    ) -> None:
        enabled = [bool(torque_states[servo_id]) for servo_id in SERVO_IDS]
        if any(enabled) and not all(enabled):
            raise SafetyError("六轴处于混合扭矩状态，拒绝自动接管")
        self.send_goals(current, speed)
        if not any(enabled):
            frame = self._sync_frame(
                TORQUE_ENABLE_ADDR,
                1,
                {servo_id: [1] for servo_id in SERVO_IDS},
            )
            self._write_frame(frame)
            time.sleep(0.25)
            after = self.torque_states()
            missing = [servo_id for servo_id, value in after.items() if not value]
            if missing:
                raise SafetyError(f"这些舵机未确认开启扭矩：{missing}")

    def takeover_current(
        self,
        *,
        speed: int = 160,
        max_drift: int = 12,
        allow_zero: bool = False,
    ) -> dict[int, int]:
        self.verify_position_modes()
        torque = self.torque_states()
        current = self.stable_sample(
            rounds=2,
            max_drift=max_drift,
            allow_zero=allow_zero,
        )
        self.establish_hold(current, speed, torque)
        return current

    def diagnostics(self) -> dict[int, dict[str, int | float]]:
        result: dict[int, dict[str, int | float]] = {}
        for servo_id in SERVO_IDS:
            torque = self.read_registers(servo_id, MAX_TORQUE_ADDR, 2)
            gains = self.read_registers(servo_id, CONTROL_GAINS_ADDR, 4)
            runtime = self.read_registers(servo_id, RUNTIME_STATUS_ADDR, 15)
            pwm = self.read_registers(servo_id, MINIMUM_PWM_ADDR, 2)
            result[servo_id] = {
                "position": runtime[4] | (runtime[5] << 8),
                "speed_raw": runtime[6] | (runtime[7] << 8),
                "load_raw": runtime[8] | (runtime[9] << 8),
                "voltage": runtime[10] / 10.0,
                "temperature": runtime[11],
                "moving": runtime[14],
                "max_torque": torque[0] | (torque[1] << 8),
                "minimum_pwm": pwm[0] | (pwm[1] << 8),
                "cw_deadband": gains[0],
                "ccw_deadband": gains[1],
                "cw_p": gains[2],
                "ccw_p": gains[3],
            }
        return result

    def move(
        self,
        goal: dict[int, int],
        *,
        label: str = "运动",
        speed: int = 160,
        arrival_tolerance: int = 5,
        settle_time: float = 0.4,
        monitor_period: float = 0.08,
        stall_timeout: float = 6.0,
        total_timeout: float = 20.0,
        trim_step: int = 2,
        trim_max: int = 48,
        trim_window: int = 30,
        trim_interval: float = 0.10,
        trim_start_delay: float = 0.25,
    ) -> dict[int, int]:
        if (
            arrival_tolerance < 1
            or settle_time < 0
            or monitor_period <= 0
            or stall_timeout <= 0
            or total_timeout <= 0
            or trim_step < 0
            or trim_max < 0
            or trim_window <= arrival_tolerance
            or trim_interval <= 0
            or trim_start_delay < 0
        ):
            raise ValueError("运动控制参数非法")
        start = self.read_positions()
        validate_positions(start, f"{label}起点")
        validate_positions(goal, f"{label}目标")
        self.send_goals(goal, speed)

        started = time.monotonic()
        last_progress_at = started
        last_trim_at = started
        arrived_since: float | None = None
        commanded = dict(goal)
        trim_active = False
        axis_steps = {servo_id: trim_step for servo_id in SERVO_IDS}
        last_sign: dict[int, int | None] = {servo_id: None for servo_id in SERVO_IDS}
        best_error = sum(abs(start[key] - goal[key]) for key in SERVO_IDS)

        while True:
            actual = self.read_positions()
            errors = {key: actual[key] - goal[key] for key in SERVO_IDS}
            total_error = sum(abs(value) for value in errors.values())
            now = time.monotonic()
            if total_error <= best_error - 2:
                best_error = total_error
                last_progress_at = now

            within = all(abs(errors[key]) <= arrival_tolerance for key in SERVO_IDS)
            if within:
                arrived_since = arrived_since or now
                if now - arrived_since >= settle_time:
                    return actual
            else:
                arrived_since = None

            max_error = max(abs(value) for value in errors.values())
            if trim_active and max_error > trim_window:
                trim_active = False
            if (
                not trim_active
                and trim_step > 0
                and max_error <= trim_window
                and now - last_progress_at >= trim_start_delay
            ):
                trim_active = True
                last_trim_at = now - trim_interval

            if trim_active and not within and now - last_trim_at >= trim_interval:
                next_command = dict(commanded)
                for servo_id in SERVO_IDS:
                    error = errors[servo_id]
                    if abs(error) <= arrival_tolerance:
                        continue
                    sign = 1 if error > 0 else -1
                    if last_sign[servo_id] is not None and last_sign[servo_id] != sign:
                        axis_steps[servo_id] = 1
                    last_sign[servo_id] = sign
                    lower, upper = SAFE_LIMITS[servo_id]
                    nominal = goal[servo_id]
                    trim_lower = max(lower, nominal - trim_max)
                    trim_upper = min(upper, nominal + trim_max)
                    direction = -1 if error > 0 else 1
                    next_command[servo_id] = min(
                        trim_upper,
                        max(
                            trim_lower,
                            commanded[servo_id] + direction * axis_steps[servo_id],
                        ),
                    )
                if next_command != commanded:
                    self.send_goals(next_command, speed)
                    commanded = next_command
                    last_trim_at = time.monotonic()

            if now - last_progress_at >= stall_timeout:
                details = ", ".join(
                    f"ID{key}:{errors[key]:+d}" for key in SERVO_IDS
                )
                raise SafetyError(f"{label}连续{stall_timeout:.1f}秒无进展：{details}")
            if now - started >= total_timeout:
                raise SafetyError(f"{label}超过总超时{total_timeout:.1f}秒")
            time.sleep(monitor_period)


def open_arm(
    port: str,
    baud: int = DEFAULT_BAUD,
    timeout: float = 0.08,
) -> CDSArm:
    """Open and return a controller; caller owns and must close it."""
    import serial

    if baud <= 0 or timeout <= 0:
        raise ValueError("baud和timeout必须大于0")
    selected = _choose_port(port)
    serial_port = serial.Serial(
        selected,
        baudrate=baud,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=min(timeout, 0.005),
        write_timeout=0.2,
    )
    time.sleep(0.2)
    serial_port.reset_input_buffer()
    return CDSArm(serial_port, timeout, selected)


@contextmanager
def connect(
    port: str = "auto",
    *,
    baud: int = DEFAULT_BAUD,
    timeout: float = 0.08,
) -> Iterator[CDSArm]:
    """Open a controller and close only the serial port on context exit."""
    arm = open_arm(port, baud, timeout)
    try:
        yield arm
    finally:
        arm.close()


validate_configuration()
