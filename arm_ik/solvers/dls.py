"""Damped least squares with limit clamping -- the default solver."""

from __future__ import annotations

import numpy as np

from ..kinematics.forward import geometric_jacobian
from ..model.transforms import FloatArray, pose_error
from . import register_solver
from .base import BaseIKSolver

__all__ = ["DLSSolver"]


@register_solver("dls")
class DLSSolver(BaseIKSolver):
    """Levenberg-Marquardt damped least squares with clamped joint limits.

    Two details matter for this arm specifically. The damping adapts per
    iteration but never drops below ``damping_min``, because a third of the
    workspace is ill-conditioned enough that an undamped step diverges. And a
    joint that would leave its range is pinned to the boundary *and* has its
    Jacobian column zeroed for the next step, so the remaining joints take over
    instead of the solver repeatedly pushing into the wall.
    """

    name = "dls"

    def _solve_from(
        self, target: FloatArray, seed: FloatArray
    ) -> tuple[FloatArray, int, list[float]]:
        cfg = self.config
        weights = cfg.weights
        q = seed.copy()
        lam = cfg.damping_init
        history: list[float] = []

        # One chain walk serves both the pose and the Jacobian, so an accepted
        # iteration costs a single FK rather than two.
        state = self.robot.chain_state(q)
        raw = pose_error(state.tip_pose, target)
        err = raw * weights
        residual = float(np.linalg.norm(err))
        history.append(residual)

        identity = np.eye(self.robot.dof, dtype=np.float64)
        iterations = 0

        for _ in range(1, cfg.max_iterations + 1):
            iterations += 1
            # Convergence is judged on the raw physical error, not the weighted
            # residual: weights steer the search, they must not redefine what
            # "close enough" means in metres and radians.
            pos_ok = float(np.linalg.norm(raw[:3])) <= cfg.position_tolerance
            rot_ok = cfg.position_only or (
                float(np.linalg.norm(raw[3:])) <= cfg.orientation_tolerance
            )
            if pos_ok and rot_ok:
                break

            jac = weights[:, None] * geometric_jacobian(state)
            # Freeze joints already pinned at a limit and pushing further out.
            jac = jac * self._free_mask(q, jac, err)[None, :]

            hessian = jac.T @ jac + (lam**2) * identity
            try:
                step = np.linalg.solve(hessian, jac.T @ err)
            except np.linalg.LinAlgError:
                lam = min(lam * cfg.damping_increase, cfg.damping_max)
                continue

            step_norm = float(np.linalg.norm(step))
            if step_norm > cfg.max_step_norm:
                step *= cfg.max_step_norm / step_norm

            candidate = np.clip(q + step, self.robot.lower, self.robot.upper)
            cand_state = self.robot.chain_state(candidate)
            cand_raw = pose_error(cand_state.tip_pose, target)
            cand_err = cand_raw * weights
            cand_residual = float(np.linalg.norm(cand_err))

            if cand_residual < residual:
                q, state, raw, err, residual = (
                    candidate,
                    cand_state,
                    cand_raw,
                    cand_err,
                    cand_residual,
                )
                lam = max(lam * cfg.damping_decrease, cfg.damping_min)
                history.append(residual)
            else:
                lam = min(lam * cfg.damping_increase, cfg.damping_max)
                if lam >= cfg.damping_max:
                    break

        return q, iterations, history

    def _free_mask(self, q: FloatArray, jac: FloatArray, err: FloatArray) -> FloatArray:
        """Zero out joints sitting on a limit whose gradient points further out.

        A joint at its upper bound is only frozen if the descent direction would
        raise it further; if the solver now wants to come back inside, the joint
        is released.
        """
        gradient = jac.T @ err
        mask = np.ones(self.robot.dof, dtype=np.float64)
        at_upper = q >= self.robot.upper - 1e-9
        at_lower = q <= self.robot.lower + 1e-9
        mask[at_upper & (gradient > 0)] = 0.0
        mask[at_lower & (gradient < 0)] = 0.0
        return mask
