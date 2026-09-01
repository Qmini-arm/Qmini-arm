"""Forward kinematics, Jacobian, and URDF-parsing tests.

The Jacobian tests matter most: an analytic Jacobian that disagrees with finite
differences is the single most common source of silent IK failure, and the error
shows up as slow convergence rather than an exception.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arm_ik import RobotModel

URDF = Path(__file__).resolve().parents[1] / "description" / "arm.urdf"


@pytest.fixture(scope="module")
def robot() -> RobotModel:
    return RobotModel.from_urdf(URDF)


@pytest.fixture(scope="module")
def samples(robot: RobotModel) -> list[np.ndarray]:
    rng = np.random.default_rng(20260901)
    return [robot.random_configuration(rng) for _ in range(40)]


def test_parses_expected_chain(robot: RobotModel) -> None:
    assert robot.dof == 6
    assert robot.joint_names[0] == "kd_base_side_to_kd_2"
    assert robot.joint_names[-1] == "kd_4_to_palm"
    assert np.all(robot.lower < robot.upper)


def test_duplicate_link_does_not_break_parse(robot: RobotModel) -> None:
    """``u3b_base`` appears twice in the URDF; the parser must keep one copy."""
    assert len(robot.urdf.links) == 17


def test_fk_is_a_valid_transform(robot: RobotModel, samples: list[np.ndarray]) -> None:
    for q in samples:
        pose = robot.fk(q)
        rot = pose[:3, :3]
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-9)
        assert np.allclose(pose[3], [0, 0, 0, 1])


def test_analytic_jacobian_matches_finite_difference(
    robot: RobotModel, samples: list[np.ndarray]
) -> None:
    worst = max(
        float(np.abs(robot.jacobian(q) - robot.numeric_jacobian(q)).max())
        for q in samples
    )
    assert worst < 1e-6, f"analytic vs numeric Jacobian differ by {worst:.2e}"


def test_home_pose_is_near_singular(robot: RobotModel) -> None:
    """The zero configuration is a boundary singularity, so IK must not seed there.

    Guards the documented reason `mid_range` is the first restart seed.
    """
    assert robot.condition_number(np.zeros(6)) > 1e3
    assert robot.condition_number(robot.mid_range) < 1e3


def test_sixth_joint_does_not_move_the_palm_origin(robot: RobotModel) -> None:
    """J6's axis passes through the tip, so its linear Jacobian column is zero."""
    for q in (np.zeros(6), robot.mid_range):
        assert np.linalg.norm(robot.jacobian(q)[:3, 5]) < 1e-9


def test_limit_helpers(robot: RobotModel) -> None:
    assert robot.within_limits(robot.mid_range)
    assert not robot.within_limits(robot.upper + 0.1)
    assert robot.within_limits(robot.clamp(robot.upper + 0.1))


def test_reach_bounds_bracket_sampled_positions(
    robot: RobotModel, samples: list[np.ndarray]
) -> None:
    lo, hi = robot.reach_bounds
    for q in samples:
        assert lo - 1e-9 <= np.linalg.norm(robot.fk(q)[:3, 3]) <= hi + 1e-9
