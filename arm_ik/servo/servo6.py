"""Independent control helpers for the palm rotation (servo6)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..model.robot_model import SERVO6_JOINT
from .mapping import JointCalibration, ServoMap

if TYPE_CHECKING:
    from ..model.robot_model import RobotModel

__all__ = ["SERVO6_JOINT", "Servo6Controller"]


class Servo6Controller:
    """Keep servo6 state separate from the five-joint position IK.

    The controller is deliberately transport-agnostic.  It validates and
    stores the wrist angle, converts it to a tick when a calibration is
    available, and composes it with a five-joint IK result.  A caller can then
    send the composed full vector through its existing bus API.
    """

    def __init__(
        self,
        robot: RobotModel,
        servo_map: ServoMap | None = None,
        *,
        angle: float | None = None,
    ) -> None:
        self.robot = robot
        self.index = robot.servo6_index
        self.joint_name = robot.joint_names[self.index]
        if self.joint_name != SERVO6_JOINT:
            raise ValueError(
                f"servo6关节名不匹配: {self.joint_name!r} != {SERVO6_JOINT!r}"
            )
        self.servo_map = servo_map
        self.calibration: JointCalibration | None = None
        if servo_map is not None:
            try:
                self.calibration = servo_map.calibrations[
                    servo_map.joint_names.index(self.joint_name)
                ]
            except ValueError as exc:
                raise ValueError(
                    f"标定中没有servo6关节{self.joint_name!r}"
                ) from exc
        model_lower, model_upper = robot.servo6_limits
        if self.calibration is None:
            self._limits = (model_lower, model_upper)
        else:
            hw_lower, hw_upper = self.calibration.radian_bounds
            self._limits = (max(model_lower, hw_lower), min(model_upper, hw_upper))
            if self._limits[0] >= self._limits[1]:
                raise ValueError("servo6的URDF限位与舵机安全窗口没有交集")
        default = float(robot.mid_range[self.index]) if angle is None else angle
        self._angle = self.validate(default)

    @property
    def limits(self) -> tuple[float, float]:
        """Effective URDF and hardware limits in radians."""
        return self._limits

    @property
    def angle(self) -> float:
        """The commanded servo6 angle in radians."""
        return self._angle

    @property
    def degrees(self) -> float:
        return float(np.degrees(self._angle))

    def validate(self, angle: float) -> float:
        value = float(angle)
        lower, upper = self._limits
        if not np.isfinite(value):
            raise ValueError("servo6角度必须是有限数")
        if not lower <= value <= upper:
            raise ValueError(
                f"servo6={value}超出限位[{lower},{upper}]"
            )
        return value

    def clamp(self, angle: float) -> float:
        """Return an angle clipped to the model's servo6 limits."""
        value = float(angle)
        if not np.isfinite(value):
            raise ValueError("servo6角度必须是有限数")
        lower, upper = self._limits
        return float(np.clip(value, lower, upper))

    def set_angle(self, angle: float, *, clamp: bool = False) -> float:
        """Set and return servo6 angle in radians.

        By default an out-of-range command is rejected.  ``clamp=True`` is
        useful for GUI slider values and makes the safety decision explicit.
        """
        self._angle = self.clamp(angle) if clamp else self.validate(angle)
        return self._angle

    def set_degrees(self, angle: float, *, clamp: bool = False) -> float:
        return self.set_angle(np.radians(float(angle)), clamp=clamp)

    def set_tick(self, tick: int, *, clamp: bool = False) -> float:
        if self.calibration is None:
            raise RuntimeError("set_tick需要提供ServoMap标定")
        value = int(tick)
        if clamp:
            value = min(max(value, self.calibration.tick_lower), self.calibration.tick_upper)
        elif not self.calibration.tick_lower <= value <= self.calibration.tick_upper:
            raise ValueError(
                f"servo6 tick={value}超出安全范围"
                f"[{self.calibration.tick_lower},{self.calibration.tick_upper}]"
            )
        return self.set_angle(self.calibration.to_radian(value), clamp=clamp)

    def to_tick(self, angle: float | None = None) -> int:
        """Convert an independent angle to its calibrated servo tick."""
        if self.calibration is None:
            raise RuntimeError("to_tick需要提供ServoMap标定")
        value = self._angle if angle is None else self.validate(angle)
        raw = self.calibration.to_tick(value)
        return min(max(raw, self.calibration.tick_lower), self.calibration.tick_upper)

    def compose(self, q_arm: npt.ArrayLike) -> np.ndarray:
        """Insert the stored servo6 angle into a five-joint IK vector."""
        return self.robot.compose_arm_q(q_arm, servo6=self._angle)
