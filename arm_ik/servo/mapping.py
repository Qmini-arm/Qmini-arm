"""Map IK joint angles to CDS55xx servo ticks and back.

The CDS55xx reports position as an integer tick in ``0..1023`` spanning 300
mechanical degrees, so one tick is ``300/1023 ≈ 0.2933`` degrees. A joint is
described by the tick that corresponds to its URDF zero pose (``center_tick``),
a sign (``direction``) and the tick span per radian.

This module deliberately does not open a serial port. Hardware access lives
behind the :class:`ServoBackend` protocol so that the kinematics stack can be
tested and simulated without a robot attached.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import yaml

from ..model.transforms import FloatArray

logger = logging.getLogger(__name__)

POSITION_MAX = 1023
DEGREES_PER_REVOLUTION = 300.0
DEGREES_PER_TICK = DEGREES_PER_REVOLUTION / POSITION_MAX
TICKS_PER_RADIAN = np.degrees(1.0) / DEGREES_PER_TICK

# Measured on the real arm: for every servo, increasing ticks move the joint in
# the URDF-positive direction. Recorded in recent.txt as "方向统一 +1".
MEASURED_DIRECTION = 1

# Above this, the URDF limits and the servo's safe window disagree enough about a
# joint's travel that the wider side is unusable, so it is worth a warning.
_FIT_RESIDUAL_WARN_DEG = 2.0


class RobotModelLike(Protocol):
    """The part of :class:`~arm_ik.RobotModel` this module needs."""

    joint_names: list[str]
    lower: FloatArray
    upper: FloatArray

    def fk(self, q: FloatArray) -> FloatArray: ...


@runtime_checkable
class ServoBackend(Protocol):
    """Minimal hardware contract the kinematics stack depends on.

    ``cds_arm.CDSArm`` satisfies this shape; any stub that returns tick dicts
    works equally well for offline testing. Only reading is required here —
    commanding the arm is ``cds_arm``'s job, not this library's.
    The command path is left to ``ServoBackend``'s implementer.
    """

    def read_positions(self) -> dict[int, int]:
        """Return the present tick of every servo, keyed by servo id."""
        ...


@dataclass(frozen=True)
class JointCalibration:
    """Per-joint conversion between URDF radians and servo ticks.

    Args:
        joint_name: URDF joint this record drives.
        servo_id: Bus id of the servo.
        center_tick: Tick observed when the joint sits at its URDF zero.
        direction: ``+1`` if increasing ticks increase the URDF joint angle,
            ``-1`` if they decrease it.
        tick_lower: Lower safety tick enforced by the hardware.
        tick_upper: Upper safety tick enforced by the hardware.
        fit_residual_deg: Disagreement between the URDF limits and the ticks
            when this record was derived. Large values mean the record is a
            guess and should be re-measured on the real arm.
    """

    joint_name: str
    servo_id: int
    center_tick: int
    direction: int
    tick_lower: int
    tick_upper: int
    fit_residual_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.direction not in (1, -1):
            raise ValueError(f"{self.joint_name}: direction必须是+1或-1")
        if not 0 <= self.tick_lower < self.tick_upper <= POSITION_MAX:
            raise ValueError(f"{self.joint_name}: tick范围非法")
        if not self.tick_lower <= self.center_tick <= self.tick_upper:
            raise ValueError(
                f"{self.joint_name}: center_tick={self.center_tick}不在"
                f"[{self.tick_lower},{self.tick_upper}]内"
            )

    def to_tick(self, angle_rad: float) -> int:
        """Convert a joint angle in radians to the nearest servo tick."""
        raw = self.center_tick + self.direction * angle_rad * TICKS_PER_RADIAN
        return int(round(raw))

    def to_radian(self, tick: int) -> float:
        """Convert a servo tick back to a joint angle in radians."""
        return self.direction * (tick - self.center_tick) / TICKS_PER_RADIAN

    @property
    def radian_bounds(self) -> tuple[float, float]:
        """Joint-space bounds implied by the hardware tick limits."""
        a = self.to_radian(self.tick_lower)
        b = self.to_radian(self.tick_upper)
        return (a, b) if a <= b else (b, a)


class ServoMap:
    """Whole-arm conversion between joint vectors and servo tick dicts."""

    def __init__(
        self,
        calibrations: Sequence[JointCalibration],
        joint_names: Sequence[str],
    ) -> None:
        by_name = {c.joint_name: c for c in calibrations}
        missing = [n for n in joint_names if n not in by_name]
        if missing:
            raise ValueError(f"标定缺少关节: {missing}")
        self.joint_names = list(joint_names)
        self.calibrations = [by_name[n] for n in self.joint_names]
        self.servo_ids = [c.servo_id for c in self.calibrations]
        if len(set(self.servo_ids)) != len(self.servo_ids):
            raise ValueError(f"servo_id重复: {self.servo_ids}")

    @property
    def dof(self) -> int:
        return len(self.calibrations)

    def to_ticks(self, q: FloatArray) -> dict[int, int]:
        """Convert a joint vector to a servo-id keyed tick dict.

        Ticks are clamped to each servo's safety window, so the result is always
        commandable. A clamp means the requested angle was outside what the
        hardware allows and is logged rather than silently absorbed.
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != self.dof:
            raise ValueError(f"q长度应为{self.dof}，收到{q.size}")
        ticks: dict[int, int] = {}
        for angle, cal in zip(q, self.calibrations, strict=True):
            raw = cal.to_tick(float(angle))
            clamped = min(max(raw, cal.tick_lower), cal.tick_upper)
            if clamped != raw:
                logger.warning(
                    "SEVERITY=HIGH %s: tick %d 超出安全范围[%d,%d]，已钳制到%d。"
                    "若这是IK结果，URDF限位与舵机安全范围不一致，需要调整",
                    cal.joint_name,
                    raw,
                    cal.tick_lower,
                    cal.tick_upper,
                    clamped,
                )
            ticks[cal.servo_id] = clamped
        return ticks

    def to_joints(self, ticks: dict[int, int]) -> FloatArray:
        """Convert a servo-id keyed tick dict to a joint vector in radians."""
        missing = [c.servo_id for c in self.calibrations if c.servo_id not in ticks]
        if missing:
            raise ValueError(f"缺少舵机读数: {missing}")
        return np.array(
            [c.to_radian(int(ticks[c.servo_id])) for c in self.calibrations],
            dtype=float,
        )

    def to_degrees(self, q: FloatArray) -> dict[int, float]:
        """Convert a joint vector to servo angles in degrees, for logging."""
        ticks = self.to_ticks(q)
        return {
            cal.servo_id: (ticks[cal.servo_id] - cal.center_tick) * DEGREES_PER_TICK
            for cal in self.calibrations
        }

    def effective_limits(
        self, robot: RobotModelLike
    ) -> tuple[FloatArray, FloatArray]:
        """Intersect the URDF limits with the hardware tick limits.

        IK should search this intersection, not the URDF box: a solution outside
        the servo's safety window gets clamped on the way to the bus, so the arm
        would silently land somewhere other than where IK said it would.
        """
        lower = np.empty(self.dof)
        upper = np.empty(self.dof)
        for i, cal in enumerate(self.calibrations):
            hw_lo, hw_hi = cal.radian_bounds
            lower[i] = max(float(robot.lower[i]), hw_lo)
            upper[i] = min(float(robot.upper[i]), hw_hi)
            if lower[i] >= upper[i]:
                raise ValueError(
                    f"{cal.joint_name}: URDF限位与舵机安全范围无交集"
                    f"(URDF[{np.degrees(robot.lower[i]):.1f},"
                    f"{np.degrees(robot.upper[i]):.1f}]° vs "
                    f"舵机[{np.degrees(hw_lo):.1f},{np.degrees(hw_hi):.1f}]°)"
                )
        return lower, upper

    def validate_against(
        self, robot: RobotModelLike, tolerance_deg: float = 2.0
    ) -> list[str]:
        """Report substantive disagreement between URDF and hardware limits.

        Sub-degree differences are expected: one tick is
        ``300/1023 = 0.293°``, and the URDF limits were themselves rounded from
        tick measurements. Only overhangs beyond ``tolerance_deg`` are reported,
        since those indicate a genuine edit on one side rather than rounding.
        """
        tol = np.radians(tolerance_deg)
        problems: list[str] = []
        for i, cal in enumerate(self.calibrations):
            hw_lo, hw_hi = cal.radian_bounds
            urdf_lo, urdf_hi = float(robot.lower[i]), float(robot.upper[i])
            overhang = max(hw_lo - urdf_lo, urdf_hi - hw_hi)
            if overhang > tol:
                problems.append(
                    f"{cal.joint_name}: URDF允许"
                    f"[{np.degrees(urdf_lo):.2f},{np.degrees(urdf_hi):.2f}]°，"
                    f"舵机只允许[{np.degrees(hw_lo):.2f},{np.degrees(hw_hi):.2f}]°，"
                    f"超出{np.degrees(overhang):.2f}°——IK将被收紧到舵机范围"
                )
            if cal.fit_residual_deg > tolerance_deg:
                problems.append(
                    f"{cal.joint_name}: 标定拟合残差{cal.fit_residual_deg:.2f}°"
                    f"偏大，center_tick或direction需要实机复核"
                )
        return problems

    @classmethod
    def from_yaml(cls, path: str | Path, joint_names: Sequence[str]) -> ServoMap:
        """Load calibration records from a YAML file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        entries = data["joints"] if isinstance(data, dict) else data
        cals = [JointCalibration(**entry) for entry in entries]
        return cls(cals, joint_names)

    def to_yaml(self, path: str | Path) -> None:
        """Write the calibration records to a YAML file."""
        payload = {
            "comment": (
                "由cds_arm.CENTER/SAFE_LIMITS与URDF限位拟合得到。"
                "fit_residual_deg偏大的条目需要实机复核。"
            ),
            "ticks_per_revolution": POSITION_MAX,
            "degrees_per_revolution": DEGREES_PER_REVOLUTION,
            "joints": [
                {
                    "joint_name": c.joint_name,
                    "servo_id": c.servo_id,
                    "center_tick": c.center_tick,
                    "direction": c.direction,
                    "tick_lower": c.tick_lower,
                    "tick_upper": c.tick_upper,
                    "fit_residual_deg": round(float(c.fit_residual_deg), 3),
                }
                for c in self.calibrations
            ],
        }
        Path(path).write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    @classmethod
    def derive(
        cls,
        robot: RobotModelLike,
        center: dict[int, int],
        safe_limits: dict[int, tuple[int, int]],
        servo_ids: Sequence[int],
        directions: dict[int, int] | None = None,
    ) -> ServoMap:
        """Build calibration records from measured servo constants.

        ``direction`` is a measured property of how each servo is mounted, so it
        is taken from ``directions`` rather than inferred. It defaults to
        :data:`MEASURED_DIRECTION` for every servo, which is what was measured on
        the real arm: increasing ticks move the joint in the URDF-positive sense.

        Earlier versions picked the sign by scoring both candidates on how well
        the tick window reproduced the URDF limits. That only worked while the
        two agreed by construction. Once a servo's safe window is widened on its
        own — as servo 2's was — the score is comparing noise, and it silently
        chose the inverted sign, which would drive the real joint backwards.

        The fit residual is still computed and stored, but purely as a
        diagnostic: a large value means the URDF limits and the servo window
        disagree about this joint's travel, and it is logged so the mismatch is
        visible instead of being absorbed into a sign flip.
        """
        cals: list[JointCalibration] = []
        for i, name in enumerate(robot.joint_names):
            sid = servo_ids[i]
            c = int(center[sid])
            lo, hi = (int(v) for v in safe_limits[sid])
            direction = MEASURED_DIRECTION if directions is None else int(directions[sid])
            urdf_lo, urdf_hi = np.degrees([robot.lower[i], robot.upper[i]])
            edges = sorted(direction * (t - c) * DEGREES_PER_TICK for t in (lo, hi))
            residual = float(max(abs(edges[0] - urdf_lo), abs(edges[1] - urdf_hi)))
            if residual > _FIT_RESIDUAL_WARN_DEG:
                logger.warning(
                    "%s: URDF限位[%.2f,%.2f]° 与舵机窗口[%.2f,%.2f]° 相差 %.2f°。"
                    "IK 会取两者交集，较宽的一侧行程用不到；"
                    "若这是刻意放宽，请同步更新 URDF 限位",
                    name,
                    urdf_lo,
                    urdf_hi,
                    edges[0],
                    edges[1],
                    residual,
                )
            cals.append(
                JointCalibration(
                    joint_name=name,
                    servo_id=sid,
                    center_tick=c,
                    direction=direction,
                    tick_lower=lo,
                    tick_upper=hi,
                    fit_residual_deg=residual,
                )
            )
        return cls(cals, robot.joint_names)


def fk_from_servo(
    robot: RobotModelLike,
    servo_map: ServoMap,
    backend: ServoBackend,
) -> tuple[FloatArray, FloatArray]:
    """Read the real arm's pose and run forward kinematics on it.

    Returns the joint vector in radians and the resulting end-effector pose.
    This is the shortest path to catching a wrong ``center_tick`` or
    ``direction``: command a known pose, read it back, and compare.
    """
    ticks = backend.read_positions()
    q = servo_map.to_joints(ticks)
    return q, robot.fk(q)

