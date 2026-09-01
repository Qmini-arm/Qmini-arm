"""Public Python API for the standalone CDS55xx arm controller."""

from .core import (
    CENTER,
    DEFAULT_BAUD,
    POSITION_MAX,
    SAFE_LIMITS,
    SERVO_IDS,
    CDSArm,
    SafetyError,
    StatusPacket,
    build_packet,
    checksum_is_valid,
    connect,
    decode_error,
    validate_configuration,
    validate_positions,
)

__version__ = "0.2.0"

__all__ = [
    "CENTER",
    "DEFAULT_BAUD",
    "POSITION_MAX",
    "SAFE_LIMITS",
    "SERVO_IDS",
    "CDSArm",
    "SafetyError",
    "StatusPacket",
    "build_packet",
    "checksum_is_valid",
    "connect",
    "decode_error",
    "validate_configuration",
    "validate_positions",
    "__version__",
]
