"""Statistics over a sampled workspace.

The analysis focuses on the numbers that matter for using a narrow-joint arm:
how far the hand reaches, how degraded the Jacobian becomes (which predicts IK
convergence trouble), and what orientation cone the wrist can cover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .sampler import SampledWorkspace

if TYPE_CHECKING:
    from ..model.robot_model import RobotModel

__all__ = ["ReachabilityReport", "analyze_workspace"]


@dataclass(frozen=True)
class ReachabilityReport:
    """Aggregate figures describing the reachable shell."""

    bounds_lower: np.ndarray
    bounds_upper: np.ndarray
    min_radius: float
    max_radius: float
    mean_radius: float
    condition_number_median: float
    condition_number_p90: float
    manipulability_median: float
    orientation_cone_half_angle_deg: float
    samples: int

    @property
    def reach_bounds(self) -> tuple[float, float]:
        return float(self.min_radius), float(self.max_radius)


_DEFAULT_REFERENCE_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def analyze_workspace(
    robot: RobotModel,
    workspace: SampledWorkspace,
    reference_axis: np.ndarray | None = None,
) -> ReachabilityReport:
    """Describe the reachable shell sampled in ``workspace``."""
    axis = _DEFAULT_REFERENCE_AXIS if reference_axis is None else np.asarray(
        reference_axis, dtype=np.float64
    )
    positions = workspace.positions
    lower = positions.min(axis=0)
    upper = positions.max(axis=0)
    radii = workspace.radii

    condition_numbers: list[float] = []
    manipulability: list[float] = []
    cone_angles: list[float] = []
    for i in range(workspace.q.shape[0]):
        condition_numbers.append(robot.condition_number(workspace.q[i]))
        manipulability.append(robot.manipulability(workspace.q[i]))
        # The wrist's z axis against the reference axis gives the orientation
        # cone half-angle this arm can sweep.
        tip_axis = workspace.poses[i, :3, 2]
        cos = float(
            np.clip(
                np.dot(tip_axis, axis) / (np.linalg.norm(tip_axis) + 1e-12),
                -1,
                1,
            )
        )
        if cos < 1e-12:
            cone_angles.append(0.0)
        else:
            cone_angles.append(np.degrees(np.arccos(cos)))

    cond = np.array(condition_numbers)
    mani = np.array(manipulability)
    cone = np.array(cone_angles)
    return ReachabilityReport(
        bounds_lower=lower,
        bounds_upper=upper,
        min_radius=float(radii.min()),
        max_radius=float(radii.max()),
        mean_radius=float(radii.mean()),
        condition_number_median=float(np.median(cond)),
        condition_number_p90=float(np.percentile(cond, 90)),
        manipulability_median=float(np.median(mani)),
        orientation_cone_half_angle_deg=float(np.percentile(cone, 95)),
        samples=int(workspace.q.shape[0]),
    )
