"""Bounded least-squares fallback built on scipy.

Slower than DLS by an order of magnitude or more, but it treats joint limits as
hard bounds and converges reliably to the nearest reachable pose when the target
is outside the workspace. Given how narrow this arm's travel is, that case comes
up often enough to justify a dedicated solver.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from ..model.transforms import FloatArray, pose_error
from . import register_solver
from .base import BaseIKSolver

__all__ = ["LeastSquaresSolver"]


@register_solver("least_squares")
class LeastSquaresSolver(BaseIKSolver):
    """Trust-region reflective least squares with joint limits as bounds."""

    name = "least_squares"

    def _solve_from(
        self, target: FloatArray, seed: FloatArray
    ) -> tuple[FloatArray, int, list[float]]:
        weights = self.config.weights
        history: list[float] = []

        def residual(q: FloatArray) -> FloatArray:
            err = pose_error(self.robot.fk(q), target) * weights
            history.append(float(np.linalg.norm(err)))
            return err

        def jacobian(q: FloatArray) -> FloatArray:
            # Sign: the residual is (target - current), so d(residual)/dq = -J.
            return -weights[:, None] * self.robot.jacobian(q)

        # Nudge the seed strictly inside the bounds; trf rejects a start exactly
        # on a bound in some scipy versions.
        span = self.robot.upper - self.robot.lower
        margin = np.minimum(1e-6, span * 1e-3)
        start = np.clip(seed, self.robot.lower + margin, self.robot.upper - margin)

        outcome = least_squares(
            residual,
            start,
            jac=jacobian,
            bounds=(self.robot.lower, self.robot.upper),
            method="trf",
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=self.config.max_iterations * 4,
        )
        return np.asarray(outcome.x, dtype=np.float64), int(outcome.nfev), history
