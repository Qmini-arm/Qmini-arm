"""Public five-finger uHand control API."""

from .core import (
    DEFAULT_BAUD,
    FINGER_COUNT,
    FINGER_NAMES,
    GESTURES,
    FingerCalibration,
    FingerValueError,
    UHand,
    UHandError,
    build_finger_packet,
    connect,
    validate_closures,
    validate_finger_angles,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_BAUD",
    "FINGER_COUNT",
    "FINGER_NAMES",
    "GESTURES",
    "FingerCalibration",
    "FingerValueError",
    "UHand",
    "UHandError",
    "build_finger_packet",
    "connect",
    "validate_closures",
    "validate_finger_angles",
    "__version__",
]
