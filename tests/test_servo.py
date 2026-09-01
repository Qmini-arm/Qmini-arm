"""Servo tick mapping tests.

A sign error here drives the real arm the wrong way, so the round-trip and
limit-reconciliation tests are the load-bearing ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arm_ik import RobotModel
from arm_ik.servo import ServoMap

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "description" / "arm.urdf"
CALIBRATION = ROOT / "arm_ik" / "config" / "servo_calibration.yaml"

# CDS55xx: half a tick is the noise floor (0.1466 deg).
HALF_TICK_DEG = 300.0 / 1023 / 2


@pytest.fixture(scope="module")
def robot() -> RobotModel:
    return RobotModel.from_urdf(URDF)


@pytest.fixture(scope="module")
def servo_map(robot: RobotModel) -> ServoMap:
    return ServoMap.from_yaml(CALIBRATION, robot.joint_names)


def test_calibration_reads_back(robot: RobotModel, servo_map: ServoMap) -> None:
    assert servo_map.dof == robot.dof
    assert servo_map.joint_names == list(robot.joint_names)
    # Each servo drives exactly one joint.
    assert len(set(servo_map.servo_ids)) == servo_map.dof


def test_round_trip_within_half_tick(
    robot: RobotModel, servo_map: ServoMap
) -> None:
    """q -> ticks -> q must return the original, up to tick quantisation.

    The bound is half a tick only when no clamping occurs. A clamp happens when
    the URDF joint bound lies slightly beyond the servo's safe tick window
    (they overlap within ~2 deg for this arm). When a joint is clamped the
    returned joint respects the servo's effective limit, so the round-trip
    error can exceed half a tick by that overlap -- but never by more than the
    window mismatch.
    """
    rng = np.random.default_rng(5)
    max_overlap = 0.0
    for cal in servo_map.calibrations:
        s_lo, s_hi = cal.radian_bounds
        joint_index = servo_map.joint_names.index(cal.joint_name)
        u_lo, u_hi = np.degrees(robot.lower[joint_index]), np.degrees(
            robot.upper[joint_index]
        )
        max_overlap = max(
            max_overlap, abs(np.degrees(s_lo) - u_lo), abs(np.degrees(s_hi) - u_hi)
        )
    # Without clamping the error is bounded by half a tick; with clamping it is
    # additionally bounded by the largest URDF/servo window overlap.
    bound = HALF_TICK_DEG + max_overlap + 1e-9
    worst = 0.0
    for _ in range(2000):
        q = robot.random_configuration(rng)
        back = servo_map.to_joints(servo_map.to_ticks(q))
        err = np.abs(np.degrees(q - back))
        assert np.all(err <= bound), f"round-trip drift {err.max():.4f} deg"
        worst = max(worst, float(err.max()))
    assert worst <= bound, f"round-trip drift {worst:.4f} deg"


def test_zero_maps_to_center_tick(servo_map: ServoMap) -> None:
    """A zero joint vector must land on each servo's calibrated center."""
    ticks = servo_map.to_ticks(np.zeros(servo_map.dof))
    for cal in servo_map.calibrations:
        assert ticks[cal.servo_id] == cal.center_tick


def test_effective_limits_narrow_urdf(robot: RobotModel, servo_map: ServoMap) -> None:
    """Reconciliation must take the stricter of URDF and servo limits."""
    lower, upper = servo_map.effective_limits(robot)
    assert np.all(lower >= robot.lower - 1e-6)
    assert np.all(upper <= robot.upper + 1e-6)
    # Exhausted ranges are signalled as a ValueError, so at least one must remain.
    assert np.all(lower < upper)


def test_servo_commands_respect_servo_window(
    robot: RobotModel, servo_map: ServoMap
) -> None:
    rng = np.random.default_rng(6)
    for _ in range(100):
        ticks = servo_map.to_ticks(robot.random_configuration(rng))
        for cal in servo_map.calibrations:
            assert cal.tick_lower <= ticks[cal.servo_id] <= cal.tick_upper


def test_mapping_reconciles_joint2_delta(
    robot: RobotModel, servo_map: ServoMap
) -> None:
    """Joint 2's URDF limit is wider than the servo window; the intersection holds."""
    lower, upper = servo_map.effective_limits(robot)
    j2 = robot.joint_names.index("kd_2_to_u3b_base")
    servo_lo = np.degrees(servo_map.calibrations[j2].radian_bounds[0])
    # The applied limit cannot exceed what the servo hardware can reach.
    assert np.degrees(lower[j2]) >= servo_lo - 1e-6
    assert np.degrees(lower[j2]) > np.degrees(robot.lower[j2]) + 1e-3
