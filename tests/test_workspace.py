"""Reachable-workspace sampling tests.

The reachability answer is what callers act on, so the tests pin down that a
sampled pose is always a legal FK result, that the shell's radius bounds agree
with the model's own cached estimate, and that clearly-unreachable points are
rejected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arm_ik import RobotModel
from arm_ik.workspace import analyze_workspace, sample_workspace

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "description" / "arm.urdf"

SAMPLES = 1500


@pytest.fixture(scope="module")
def robot() -> RobotModel:
    return RobotModel.from_urdf(URDF)


@pytest.fixture(scope="module")
def workspace(robot: RobotModel):
    return sample_workspace(robot, count=SAMPLES, seed=0)


def test_every_sample_is_a_legal_configuration(robot: RobotModel, workspace) -> None:
    assert workspace.q.shape == (SAMPLES, robot.dof)
    assert workspace.poses.shape == (SAMPLES, 4, 4)
    assert robot.within_limits(workspace.q)


def test_sampled_poses_match_forward_kinematics(robot: RobotModel, workspace) -> None:
    """The stored pose must be the FK of the stored joint vector, not a stale copy."""
    for i in (0, SAMPLES // 2, SAMPLES - 1):
        assert np.allclose(robot.fk(workspace.q[i]), workspace.poses[i], atol=1e-12)


def test_sampling_is_reproducible(robot: RobotModel) -> None:
    a = sample_workspace(robot, count=200, seed=7)
    b = sample_workspace(robot, count=200, seed=7)
    assert np.allclose(a.q, b.q)
    assert not np.allclose(a.q, sample_workspace(robot, count=200, seed=8).q)


def test_reachability_accepts_sampled_points(workspace) -> None:
    """A point taken from the shell itself must be reported reachable."""
    for i in (0, SAMPLES // 3, SAMPLES - 1):
        assert workspace.is_reachable(workspace.positions[i], tol=1e-6)


def test_reachability_rejects_far_points(workspace) -> None:
    assert not workspace.is_reachable([1.5, 0.0, 0.0])
    assert not workspace.is_reachable([0.0, 0.0, -1.0])


def test_radii_agree_with_model_reach_bounds(robot: RobotModel, workspace) -> None:
    """Sampled radii must sit inside the model's own (deliberately widened) bounds."""
    low, high = robot.reach_bounds
    assert workspace.radii.min() >= low
    assert workspace.radii.max() <= high


def test_report_bounds_enclose_all_samples(robot: RobotModel, workspace) -> None:
    report = analyze_workspace(robot, workspace)
    assert report.samples == SAMPLES
    positions = workspace.positions
    assert np.all(positions >= report.bounds_lower - 1e-12)
    assert np.all(positions <= report.bounds_upper + 1e-12)
    assert report.min_radius <= report.mean_radius <= report.max_radius


def test_report_records_the_narrow_workspace(robot: RobotModel, workspace) -> None:
    """This arm reaches a thin shell, not a solid ball, and is often ill-conditioned.

    These are the numbers that explain why IK needs restarts and why a target can
    be position-reachable but not pose-reachable.
    """
    report = analyze_workspace(robot, workspace)
    # The shell never collapses to a point nor extends past the link lengths.
    assert 0.01 < report.min_radius < report.max_radius < 0.5
    # A well-conditioned 6-DoF arm would sit near single digits; this one does not.
    assert report.condition_number_p90 > 100.0
    assert report.manipulability_median > 0.0
    assert 0.0 <= report.orientation_cone_half_angle_deg <= 180.0


def test_reference_axis_changes_the_cone(robot: RobotModel, workspace) -> None:
    """The cone is measured against a caller-supplied axis and must respond to it."""
    up = analyze_workspace(robot, workspace, reference_axis=np.array([0.0, 0.0, 1.0]))
    down = analyze_workspace(robot, workspace, reference_axis=np.array([0.0, 0.0, -1.0]))
    assert not np.isclose(
        up.orientation_cone_half_angle_deg, down.orientation_cone_half_angle_deg
    )
