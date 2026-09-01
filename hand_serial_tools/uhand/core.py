"""Five-finger-only direct USB control for the uHand UNO.

The legacy uHand firmware accepts a six-byte servo payload.  This module
intentionally exposes only five fingers; the sixth protocol byte is always a
fixed compatibility placeholder and is never accepted from callers.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_COUNT = len(FINGER_NAMES)
DEFAULT_BAUD = 115200
_RESERVED_PROTOCOL_VALUE = 90


class UHandError(RuntimeError):
    """Base exception for uHand connection and transmission failures."""


class FingerValueError(ValueError):
    """Raised when a finger target, closure, or gesture is invalid."""


def _validate_numeric_values(
    values: Sequence[float],
    *,
    label: str,
    low: float,
    high: float,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)) or len(values) != FINGER_COUNT:
        raise FingerValueError(
            f"{label} must contain exactly five values in order: "
            "thumb, index, middle, ring, pinky"
        )

    checked: list[float] = []
    for name, value in zip(FINGER_NAMES, values):
        if isinstance(value, bool):
            raise FingerValueError(f"{label}.{name} must be numeric, not bool")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise FingerValueError(f"{label}.{name} must be numeric") from exc
        if not math.isfinite(numeric) or not low <= numeric <= high:
            raise FingerValueError(
                f"{label}.{name} must be within [{low:g}, {high:g}], got {value!r}"
            )
        checked.append(numeric)
    return tuple(checked)


def validate_finger_angles(finger_angles: Sequence[float]) -> tuple[int, ...]:
    """Validate and round H1..H5 logical angles without silently clamping."""

    checked = _validate_numeric_values(
        finger_angles,
        label="finger_angles",
        low=0.0,
        high=180.0,
    )
    return tuple(int(round(value)) for value in checked)


def validate_closures(closures: Sequence[float]) -> tuple[float, ...]:
    """Validate five normalized closures, where 0=open and 1=closed."""

    return _validate_numeric_values(
        closures,
        label="closures",
        low=0.0,
        high=1.0,
    )


@dataclass(frozen=True)
class FingerCalibration:
    """Per-finger logical endpoints used to map normalized closure to angles."""

    open_angles: tuple[int, ...] = (180, 180, 180, 180, 180)
    closed_angles: tuple[int, ...] = (0, 0, 0, 0, 0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "open_angles", validate_finger_angles(self.open_angles))
        object.__setattr__(self, "closed_angles", validate_finger_angles(self.closed_angles))

    def angles_for_closures(self, closures: Sequence[float]) -> tuple[int, ...]:
        values = validate_closures(closures)
        return tuple(
            int(round(open_angle + closure * (closed_angle - open_angle)))
            for open_angle, closed_angle, closure in zip(
                self.open_angles, self.closed_angles, values
            )
        )


# A preset is a normalized closure vector in thumb/index/middle/ring/pinky order.
# 0 means fully open; 1 means fully closed.  "power" is the task-level name for
# a fist-like enveloping grasp and intentionally aliases "fist".
GESTURES: Mapping[str, tuple[float, ...]] = MappingProxyType(
    {
        "open": (0.0, 0.0, 0.0, 0.0, 0.0),
        "fist": (1.0, 1.0, 1.0, 1.0, 1.0),
        "power": (1.0, 1.0, 1.0, 1.0, 1.0),
        "point": (1.0, 0.0, 1.0, 1.0, 1.0),
        "victory": (1.0, 0.0, 0.0, 1.0, 1.0),
        "thumbs_up": (0.0, 1.0, 1.0, 1.0, 1.0),
        "pinch": (1.0, 1.0, 0.0, 0.0, 0.0),
        "tripod": (1.0, 1.0, 1.0, 0.0, 0.0),
    }
)


def build_finger_packet(finger_angles: Sequence[float]) -> bytes:
    """Build the legacy AA77 frame from exactly H1..H5 logical angles.

    The firmware-compatible sixth byte is fixed internally.  Supplying six
    values is rejected so an arm wrist target cannot cross this API boundary.
    """

    fingers = validate_finger_angles(finger_angles)
    payload = bytes([0x01, 0x06, *fingers, _RESERVED_PROTOCOL_VALUE])
    checksum = (~sum(payload)) & 0xFF
    return b"\xAA\x77" + payload + bytes([checksum])


def _discover_unique_port() -> str:
    try:
        from serial.tools import list_ports  # type: ignore
    except ImportError as exc:
        raise UHandError("pyserial is required; install the uhand-control package") from exc

    candidates = [item.device for item in list_ports.comports()]
    if not candidates:
        raise UHandError("no USB serial port found; pass the uHand port explicitly")

    # Prefer common UNO Type-B CDC names and do not auto-select adapter-style
    # serial devices. A single Windows COM candidate is accepted because COM
    # names do not expose USB device type; multiple candidates require an
    # explicit port.
    adapter_tokens = ("usbserial", "ftdi", "ch340", "ttyusb")
    direct_candidates = [
        port
        for port in candidates
        if any(token in port.lower() for token in ("usbmodem", "ttyacm", "arduino"))
    ]
    if len(direct_candidates) == 1:
        return direct_candidates[0]
    if (
        not direct_candidates
        and len(candidates) == 1
        and not any(token in candidates[0].lower() for token in adapter_tokens)
    ):
        return candidates[0]
    choices = direct_candidates or candidates
    joined = ", ".join(choices)
    raise UHandError(
        f"could not uniquely identify a direct Type-B USB port ({joined}); "
        "pass the uHand USB port explicitly"
    )


class UHand:
    """Five-finger controller backed by a latest-command-only writer thread."""

    def __init__(
        self,
        serial_port: Any,
        *,
        port: str,
        calibration: FingerCalibration | None = None,
    ) -> None:
        self._serial = serial_port
        self.port = port
        self.baud = DEFAULT_BAUD
        self.calibration = calibration or FingerCalibration()

        self._condition = threading.Condition()
        self._pending: tuple[int, bytes] | None = None
        self._next_sequence = 0
        self._sent_sequence = 0
        self._writer_error: Exception | None = None
        self._stopping = False
        self._closed = False
        self._motion_lock = threading.Lock()
        self._last_target = self.calibration.open_angles
        self._writer_thread = threading.Thread(
            target=self._write_loop,
            name="uHand-five-finger-writer",
            daemon=True,
        )
        self._writer_thread.start()

    @property
    def last_target(self) -> tuple[int, ...]:
        """Last requested H1..H5 target; this is not physical position feedback."""

        with self._condition:
            return self._last_target

    @property
    def available_gestures(self) -> tuple[str, ...]:
        return tuple(GESTURES)

    def _raise_if_unavailable(self) -> None:
        if self._closed:
            raise UHandError("uHand connection is closed")
        if self._writer_error is not None:
            raise UHandError(f"serial writer failed: {self._writer_error}") from self._writer_error

    def _queue(self, angles: Sequence[float]) -> int:
        target = validate_finger_angles(angles)
        packet = build_finger_packet(target)
        with self._condition:
            self._raise_if_unavailable()
            self._next_sequence += 1
            sequence = self._next_sequence
            # Replacing an unsent packet is deliberate: real-time control should
            # execute the newest pose instead of accumulating stale poses.
            self._pending = (sequence, packet)
            self._last_target = target
            self._condition.notify_all()
            return sequence

    def _wait_sent(self, sequence: int, timeout: float = 1.0) -> None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._sent_sequence < sequence:
                self._raise_if_unavailable()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise UHandError("timed out waiting for the finger command to be written")
                self._condition.wait(remaining)

    def _write_loop(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                sequence, packet = self._pending
                self._pending = None
            try:
                written = self._serial.write(packet)
                if written is not None and written != len(packet):
                    raise OSError(f"short serial write: {written}/{len(packet)} bytes")
            except Exception as exc:  # pyserial exception classes vary by platform
                with self._condition:
                    self._writer_error = exc
                    self._pending = None
                    self._condition.notify_all()
                return
            with self._condition:
                self._sent_sequence = max(self._sent_sequence, sequence)
                self._condition.notify_all()

    def command_fingers(
        self,
        finger_angles: Sequence[float],
        *,
        wait: bool = False,
        timeout: float = 1.0,
    ) -> tuple[int, ...]:
        """Queue one H1..H5 target, replacing any older unsent target.

        This is the low-latency method for an external control loop.  Use
        :meth:`set_fingers` when the library should interpolate the motion.
        """

        target = validate_finger_angles(finger_angles)
        sequence = self._queue(target)
        if wait:
            self._wait_sent(sequence, timeout)
        return target

    def set_fingers(
        self,
        finger_angles: Sequence[float],
        *,
        duration: float = 0.0,
        rate_hz: float = 30.0,
    ) -> tuple[int, ...]:
        """Move all five fingers to logical angles, optionally interpolated."""

        target = validate_finger_angles(finger_angles)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration must be a finite value >= 0")
        if not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("rate_hz must be a finite value > 0")

        with self._motion_lock:
            start = self.last_target
            if duration == 0:
                sequence = self._queue(target)
                self._wait_sent(sequence)
                return target

            steps = max(1, int(math.ceil(duration * rate_hz)))
            started = time.monotonic()
            final_sequence = 0
            for step in range(1, steps + 1):
                ratio = step / steps
                pose = tuple(
                    int(round(old + ratio * (new - old)))
                    for old, new in zip(start, target)
                )
                final_sequence = self._queue(pose)
                deadline = started + step * duration / steps
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            self._wait_sent(final_sequence)
            return target

    def set_finger(
        self,
        finger: str,
        angle: float,
        *,
        duration: float = 0.0,
        rate_hz: float = 30.0,
    ) -> tuple[int, ...]:
        """Move one named finger while preserving the other requested targets."""

        normalized = finger.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "h1": "thumb",
            "h2": "index",
            "h3": "middle",
            "h4": "ring",
            "h5": "pinky",
            "little": "pinky",
            "little_finger": "pinky",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in FINGER_NAMES:
            raise FingerValueError(
                f"unknown finger {finger!r}; expected one of {', '.join(FINGER_NAMES)}"
            )
        checked = validate_finger_angles(
            tuple(
                angle if name == normalized else value
                for name, value in zip(FINGER_NAMES, self.last_target)
            )
        )
        return self.set_fingers(checked, duration=duration, rate_hz=rate_hz)

    def set_closures(
        self,
        closures: Sequence[float],
        *,
        duration: float = 0.0,
        rate_hz: float = 30.0,
    ) -> tuple[int, ...]:
        """Move five fingers using normalized closure values (0=open, 1=closed)."""

        target = self.calibration.angles_for_closures(closures)
        return self.set_fingers(target, duration=duration, rate_hz=rate_hz)

    def gesture(
        self,
        name: str,
        *,
        amount: float = 1.0,
        duration: float = 0.5,
        rate_hz: float = 30.0,
    ) -> tuple[int, ...]:
        """Execute a calibrated preset gesture.

        ``amount`` scales each curled finger from open (0) to the preset (1).
        """

        key = name.strip().lower().replace("-", "_").replace(" ", "_")
        if key not in GESTURES:
            raise FingerValueError(
                f"unknown gesture {name!r}; available: {', '.join(GESTURES)}"
            )
        try:
            numeric_amount = float(amount)
        except (TypeError, ValueError) as exc:
            raise FingerValueError("gesture amount must be numeric within [0, 1]") from exc
        if (
            isinstance(amount, bool)
            or not math.isfinite(numeric_amount)
            or not 0 <= numeric_amount <= 1
        ):
            raise FingerValueError("gesture amount must be within [0, 1]")
        closures = tuple(value * numeric_amount for value in GESTURES[key])
        return self.set_closures(closures, duration=duration, rate_hz=rate_hz)

    def set_grasp(
        self,
        grasp_type: str,
        *,
        closure: float = 1.0,
        duration: float = 0.5,
        rate_hz: float = 30.0,
    ) -> tuple[int, ...]:
        """Task-level alias for open/power/pinch/tripod grasp presets."""

        key = grasp_type.strip().lower().replace("-", "_").replace(" ", "_")
        if key not in {"open", "power", "pinch", "tripod"}:
            raise FingerValueError("grasp_type must be open, power, pinch, or tripod")
        return self.gesture(
            key,
            amount=closure,
            duration=duration,
            rate_hz=rate_hz,
        )

    def close(self) -> None:
        """Close the serial connection; no finger or arm target is changed."""

        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._stopping = True
            self._pending = None
            self._condition.notify_all()
        self._writer_thread.join(timeout=0.5)
        self._serial.close()

    def __enter__(self) -> "UHand":
        self._raise_if_unavailable()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def connect(
    port: str = "auto",
    *,
    timeout: float = 0.0,
    write_timeout: float = 0.1,
    startup_delay: float = 2.0,
    calibration: FingerCalibration | None = None,
) -> UHand:
    """Open a direct Type-B USB link to a five-finger-only controller."""

    if port == "auto":
        port = _discover_unique_port()
    if not math.isfinite(startup_delay) or startup_delay < 0:
        raise ValueError("startup_delay must be a finite value >= 0")

    try:
        import serial  # type: ignore
    except ImportError as exc:
        raise UHandError("pyserial is required; install the uhand-control package") from exc

    serial_port = serial.Serial(
        port,
        baudrate=DEFAULT_BAUD,
        timeout=timeout,
        write_timeout=write_timeout,
    )
    try:
        if startup_delay:
            time.sleep(startup_delay)
        return UHand(
            serial_port,
            port=port,
            calibration=calibration,
        )
    except Exception:
        serial_port.close()
        raise
