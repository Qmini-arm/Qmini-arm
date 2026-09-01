"""Reachable-workspace sampling.

This arm's joint travel is very narrow, so the reachable set is a thin shell
rather than a solid volume. Sampling it is the most direct way to answer "can
the arm reach that pose?" without hand-computing the workspace boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..model.transforms import FloatArray

if TYPE_CHECKING:
    from ..model.robot_model import RobotModel

__all__ = ["SampledWorkspace", "sample_workspace"]


class SampledWorkspace:
    """The result of sampling: end-effector poses plus per-sample reachability."""

    def __init__(self, robot: RobotModel, q: FloatArray, poses: FloatArray) -> None:
        self.robot = robot
        self.q = q
        self.poses = poses

    @property
    def positions(self) -> FloatArray:
        return self.poses[:, :3, 3].copy()

    @property
    def orientations(self) -> FloatArray:
        return self.poses[:, :3, :3].copy()

    @property
    def radii(self) -> FloatArray:
        return np.linalg.norm(self.positions, axis=1)

    def is_reachable(self, target: npt.ArrayLike, tol: float = 1e-2) -> bool:
        """Whether a target position lies within the sampled reachable shell.

        Points the arm cannot physically reach are excluded, so this is stricter
        than a plain distance test against the sampling bounding box.
        """
        p = np.asarray(target, dtype=np.float64).reshape(3)
        distances = np.linalg.norm(self.positions - p, axis=1)
        return bool(float(distances.min()) <= tol)


def sample_workspace(
    robot: RobotModel,
    count: int = 20000,
    seed: int = 0,
) -> SampledWorkspace:
    """Uniformly sample joint space and run FK for each sample.

    ``count`` samples of a narrow joint box cheaply cover the reachable shell.
    """
    rng = np.random.default_rng(seed)
    q_all = np.empty((count, robot.dof), dtype=np.float64)
    for i in range(count):
        q_all[i] = robot.random_configuration(rng)
    poses = np.array([robot.fk(q_all[i]) for i in range(count)], dtype=np.float64)
    return SampledWorkspace(robot, q_all, poses)
