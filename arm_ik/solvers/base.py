"""Solver base class and shared seed/classification logic."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping

import numpy as np
import numpy.typing as npt

from ..config import IKResult, IKStatus, SolverConfig
from ..model.robot_model import RobotModel
from ..model.transforms import FloatArray, pose_error

logger = logging.getLogger(__name__)

__all__ = ["BaseIKSolver"]


class BaseIKSolver(ABC):
    """Common scaffolding: seed ordering, error measurement, status assignment."""

    name: str = "base"

    def __init__(self, robot: RobotModel, config: SolverConfig | None = None) -> None:
        self.robot = robot
        self.config = config or SolverConfig()
        # A position solve can hold a wrist joint fixed while the same full
        # configuration vector is retained for FK and servo output.  The mask
        # lives on the solver instance so custom solvers remain source
        # compatible with the existing ``_solve_from(target, seed)`` hook.
        self._fixed_mask = np.zeros(robot.dof, dtype=bool)
        self._fixed_values = np.zeros(robot.dof, dtype=np.float64)

    @abstractmethod
    def _solve_from(
        self, target: FloatArray, seed: FloatArray
    ) -> tuple[FloatArray, int, list[float]]:
        """Run one attempt; return the configuration, iterations, and residuals."""

    def solve(
        self,
        target: npt.ArrayLike,
        seed: npt.ArrayLike | None = None,
        *,
        fixed_joints: Mapping[int | str, float] | None = None,
    ) -> IKResult:
        """Solve with multiple restarts, returning the best attempt.

        Attempts stop early on convergence. When nothing converges the attempt
        with the smallest weighted residual is returned, which is what makes an
        unreachable target yield a useful nearest pose.
        """
        goal = np.asarray(target, dtype=np.float64)
        if goal.shape != (4, 4):
            raise ValueError(f"目标位姿必须是4x4变换矩阵，得到shape={goal.shape}")

        self._set_fixed_joints(fixed_joints)

        best: IKResult | None = None
        for attempt, start in enumerate(self._seeds(seed), start=1):
            if np.any(self._fixed_mask):
                start = start.copy()
                start[self._fixed_mask] = self._fixed_values[self._fixed_mask]
            q, iterations, history = self._solve_from(goal, start)
            q = self.robot.clamp(q)
            if np.any(self._fixed_mask):
                # Enforce the contract even for third-party solvers that do not
                # inspect ``_active_mask`` themselves.
                q[self._fixed_mask] = self._fixed_values[self._fixed_mask]
            result = self._classify(goal, q, iterations, attempt, history)
            if best is None or self._is_better(result, best):
                best = result
            if result.status is IKStatus.CONVERGED:
                break
        assert best is not None  # at least one seed is always produced
        logger.debug("%s: %s", self.name, best)
        return best

    def _set_fixed_joints(
        self, fixed_joints: Mapping[int | str, float] | None
    ) -> None:
        """Validate and install optional fixed joint values for one solve."""
        self._fixed_mask.fill(False)
        self._fixed_values.fill(0.0)
        if fixed_joints is None:
            return
        for key, value in fixed_joints.items():
            if isinstance(key, str):
                try:
                    index = self.robot.joint_names.index(key)
                except ValueError as exc:
                    raise ValueError(f"固定关节不存在: {key!r}") from exc
            else:
                index = int(key)
                if not 0 <= index < self.robot.dof:
                    raise ValueError(f"固定关节索引越界: {index}")
            angle = float(value)
            if not np.isfinite(angle):
                raise ValueError(f"固定关节角必须是有限数: {key!r}")
            if not self.robot.lower[index] <= angle <= self.robot.upper[index]:
                name = self.robot.joint_names[index]
                raise ValueError(
                    f"固定关节{name}={angle}超出限位"
                    f"[{self.robot.lower[index]},{self.robot.upper[index]}]"
                )
            self._fixed_mask[index] = True
            self._fixed_values[index] = angle

    @property
    def _active_mask(self) -> np.ndarray:
        """Boolean mask of joints the numerical solver is allowed to change."""
        return ~self._fixed_mask

    def _seeds(self, seed: npt.ArrayLike | None) -> Iterator[FloatArray]:
        """Yield start configurations, most promising first.

        Mid-travel comes before the caller's seed only when no seed is given.
        The zero configuration is never offered on its own: for this arm it is a
        fully-extended boundary singularity.
        """
        if seed is not None:
            yield self.robot.clamp(seed)
        yield self.robot.mid_range
        rng = np.random.default_rng(self.config.seed)
        for _ in range(max(0, self.config.restarts - (2 if seed is not None else 1))):
            yield self.robot.random_configuration(rng)

    def _errors(self, target: FloatArray, q: FloatArray) -> tuple[float, float]:
        err = pose_error(self.robot.fk(q), target)
        return float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))

    def _weighted_residual(self, target: FloatArray, q: FloatArray) -> float:
        err = pose_error(self.robot.fk(q), target) * self.config.weights
        return float(np.linalg.norm(err))

    def _classify(
        self,
        target: FloatArray,
        q: FloatArray,
        iterations: int,
        attempt: int,
        history: list[float],
    ) -> IKResult:
        pos_err, rot_err = self._errors(target, q)
        cfg = self.config
        rot_ok = cfg.position_only or rot_err <= cfg.orientation_tolerance
        if pos_err <= cfg.position_tolerance and rot_ok:
            status = IKStatus.CONVERGED
        elif pos_err <= cfg.position_tolerance:
            status = IKStatus.POSITION_ONLY
        elif self._outside_reach(target):
            # Distinguished from LIMIT_BLOCKED because the caller's remedy
            # differs: move the target closer rather than relax orientation.
            status = IKStatus.OUT_OF_REACH
        elif self._at_limits(q):
            status = IKStatus.LIMIT_BLOCKED
        elif self.robot.condition_number(q) > 1e6:
            status = IKStatus.SINGULAR
        else:
            status = IKStatus.MAX_ITER
        return IKResult(
            status=status,
            q=q,
            position_error=pos_err,
            orientation_error=rot_err,
            iterations=iterations,
            restarts_used=attempt - 1,
            residual_history=tuple(history),
        )

    def _outside_reach(self, target: FloatArray) -> bool:
        """Whether the target position is provably outside the reachable shell."""
        radius = float(np.linalg.norm(target[:3, 3]))
        low, high = self.robot.reach_bounds
        return radius < low - 1e-6 or radius > high + 1e-6

    def _at_limits(self, q: FloatArray, tol: float = 1e-6) -> bool:
        """Whether any joint is pinned against a limit."""
        return bool(
            np.any(q <= self.robot.lower + tol) or np.any(q >= self.robot.upper - tol)
        )

    @staticmethod
    def _is_better(candidate: IKResult, incumbent: IKResult) -> bool:
        """Rank by status tier first, then by position error.

        Position is ranked ahead of orientation because a reachable position with
        imperfect orientation is usually actionable, while the reverse is not.
        """
        if candidate.status.rank != incumbent.status.rank:
            return candidate.status.rank < incumbent.status.rank
        if abs(candidate.position_error - incumbent.position_error) > 1e-12:
            return candidate.position_error < incumbent.position_error
        return candidate.orientation_error < incumbent.orientation_error
