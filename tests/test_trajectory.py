"""Trajectory generation tests.

The load-bearing properties are that a path never leaves the joint limits, that
its endpoints land exactly on what was asked for, and that the stated velocity
and acceleration limits actually constrain the result rather than being
accepted and ignored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arm_ik import RobotModel
from arm_ik.trajectory import (
    interpolate_cartesian,
    interpolate_joint,
    time_parameterize,
)
from arm_ik.trajectory.timing import _limit_acceleration

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "description" / "arm.urdf"


@pytest.fixture(scope="module")
def robot() -> RobotModel:
    return RobotModel.from_urdf(URDF)


def test_joint_path_hits_endpoints_exactly(robot: RobotModel) -> None:
    q0 = robot.mid_range
    q1 = robot.clamp(robot.lower + 0.3)
    path = interpolate_joint(q0, q1, points=40)
    assert path.shape == (40, robot.dof)
    assert np.allclose(path[0], q0)
    assert np.allclose(path[-1], q1)


def test_joint_path_stays_within_limits(robot: RobotModel) -> None:
    """Interpolating between two legal configurations cannot leave the box."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        q0 = robot.random_configuration(rng)
        q1 = robot.random_configuration(rng)
        assert robot.within_limits(interpolate_joint(q0, q1, points=25))


def test_joint_path_starts_and_ends_at_rest(robot: RobotModel) -> None:
    """The quintic ease means the first and last steps are far smaller than mid-path."""
    path = interpolate_joint(robot.mid_range, robot.clamp(robot.lower + 0.3), points=60)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert steps[0] < steps[len(steps) // 2] / 10
    assert steps[-1] < steps[len(steps) // 2] / 10


def test_cartesian_path_solves_between_reachable_poses(robot: RobotModel) -> None:
    q_a = robot.mid_range
    q_b = robot.clamp(robot.mid_range + np.array([0.05, -0.03, 0.1, 0.04, -0.02, 0.03]))
    q_path, pose_path = interpolate_cartesian(robot, robot.fk(q_a), robot.fk(q_b), points=15)
    assert q_path.shape == (15, robot.dof)
    assert pose_path.shape == (15, 4, 4)
    assert robot.within_limits(q_path)
    # The IK solution at each end must reproduce the pose that was asked for.
    assert np.allclose(robot.fk(q_path[0])[:3, 3], robot.fk(q_a)[:3, 3], atol=1e-4)
    assert np.allclose(robot.fk(q_path[-1])[:3, 3], robot.fk(q_b)[:3, 3], atol=1e-4)


def test_cartesian_path_reports_unreachable_waypoint(robot: RobotModel) -> None:
    """A straight line through unreachable space must fail loudly, not silently clamp.

    The zero pose's orientation cannot be held at the mid-range position, so the
    line between them leaves the reachable set partway along.
    """
    with pytest.raises(ValueError, match="无法求解IK"):
        interpolate_cartesian(
            robot, robot.fk(np.zeros(robot.dof)), robot.fk(robot.mid_range), points=20
        )


def test_time_parameterize_respects_velocity_limits(robot: RobotModel) -> None:
    path = interpolate_joint(robot.mid_range, robot.clamp(robot.lower + 0.3), points=30)
    limits = np.full(robot.dof, 2.0)
    timed = time_parameterize(path, limits, dt=0.02)
    assert np.all(np.abs(timed.qd) <= limits + 1e-9)
    assert timed.duration > 0.0
    assert timed.q.shape[1] == robot.dof
    # The final sample must land on the final waypoint, at rest.
    assert np.allclose(timed.q[-1], path[-1])
    assert np.allclose(timed.qd[-1], 0.0)


def test_tighter_velocity_limit_takes_longer(robot: RobotModel) -> None:
    path = interpolate_joint(robot.mid_range, robot.clamp(robot.lower + 0.3), points=30)
    fast = time_parameterize(path, np.full(robot.dof, 2.0), dt=0.02)
    slow = time_parameterize(path, np.full(robot.dof, 0.5), dt=0.02)
    assert slow.duration > fast.duration


def test_accel_limit_is_enforced_at_waypoints(robot: RobotModel) -> None:
    """``accel`` must actually bind, not be accepted and dropped.

    The guarantee is per-waypoint: the velocity change where two segments meet,
    over their mean duration, stays within the limit. It is deliberately not a
    bound on ``diff(qd) / dt`` of the returned grid.
    """
    q_a = robot.mid_range
    q_b = robot.clamp(robot.mid_range + np.array([0.3, 0.0, 0.4, 0.0, 0.0, 0.0]))
    q_c = robot.clamp(robot.mid_range + np.array([-0.3, 0.0, -0.4, 0.0, 0.0, 0.0]))
    # A sharp corner at q_b gives the acceleration limit something to do.
    path = np.vstack(
        [interpolate_joint(q_a, q_b, points=8), interpolate_joint(q_b, q_c, points=8)]
    )
    limits = np.full(robot.dof, 2.0)
    deltas = np.diff(path, axis=0)

    for accel in (5.0, 1.0):
        base = np.array(
            [max(float(np.max(np.abs(d) / limits)), 0.02) for d in deltas]
        )
        stretched = _limit_acceleration(deltas, base, accel)
        worst = 0.0
        for i in range(len(stretched) - 1):
            v_in = deltas[i] / stretched[i]
            v_out = deltas[i + 1] / stretched[i + 1]
            window = 0.5 * (stretched[i] + stretched[i + 1])
            worst = max(worst, float(np.max(np.abs(v_out - v_in))) / window)
        assert worst <= accel + 1e-6, f"accel={accel} 未被约束: {worst:.3f}"


def test_accel_limit_slows_the_trajectory(robot: RobotModel) -> None:
    q_a = robot.mid_range
    q_b = robot.clamp(robot.mid_range + np.array([0.3, 0.0, 0.4, 0.0, 0.0, 0.0]))
    q_c = robot.clamp(robot.mid_range + np.array([-0.3, 0.0, -0.4, 0.0, 0.0, 0.0]))
    path = np.vstack(
        [interpolate_joint(q_a, q_b, points=8), interpolate_joint(q_b, q_c, points=8)]
    )
    limits = np.full(robot.dof, 2.0)
    free = time_parameterize(path, limits, dt=0.02)
    eased = time_parameterize(path, limits, dt=0.02, accel=1.0)
    assert eased.duration > free.duration
    # Velocity limits still hold once acceleration stretching is applied.
    assert np.all(np.abs(eased.qd) <= limits + 1e-9)


def test_time_parameterize_rejects_bad_input(robot: RobotModel) -> None:
    path = interpolate_joint(robot.mid_range, robot.clamp(robot.lower + 0.3), points=10)
    with pytest.raises(ValueError, match="长度必须等于dof"):
        time_parameterize(path, np.full(robot.dof - 1, 2.0))
    with pytest.raises(ValueError, match="必须是一维"):
        time_parameterize(path, np.full((robot.dof, 2), 2.0))
    with pytest.raises(ValueError, match="至少需要两个路点"):
        time_parameterize(path[:1], np.full(robot.dof, 2.0))
    with pytest.raises(ValueError, match="accel必须为正"):
        time_parameterize(path, np.full(robot.dof, 2.0), accel=0.0)


def test_time_parameterize_accepts_list_limits(robot: RobotModel) -> None:
    """A plain list must work; it used to raise ``AttributeError`` on ``.ndim``."""
    path = interpolate_joint(robot.mid_range, robot.clamp(robot.lower + 0.3), points=10)
    timed = time_parameterize(path, [2.0] * robot.dof, dt=0.02)
    assert timed.duration > 0.0


def test_sample_at_returns_a_configuration(robot: RobotModel) -> None:
    path = interpolate_joint(robot.mid_range, robot.clamp(robot.lower + 0.3), points=20)
    timed = time_parameterize(path, np.full(robot.dof, 2.0), dt=0.02)
    mid = timed.sample_at(timed.duration / 2)
    assert mid.shape == (robot.dof,)
    assert robot.within_limits(mid)
    # Clamping at both ends rather than raising.
    assert timed.sample_at(-1.0).shape == (robot.dof,)
    assert timed.sample_at(timed.duration * 10).shape == (robot.dof,)
