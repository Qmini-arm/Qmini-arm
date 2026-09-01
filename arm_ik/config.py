"""Immutable configuration and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .model.transforms import FloatArray

__all__ = ["IKStatus", "SolverConfig", "IKResult"]


class IKStatus(Enum):
    """Why a solve ended.

    A boolean would not be enough for this arm: its joint travel is narrow
    enough that partial success is the common case, and callers need to tell
    "close enough in position, orientation unreachable" apart from "diverged".
    """

    CONVERGED = "converged"
    POSITION_ONLY = "position_only"
    LIMIT_BLOCKED = "limit_blocked"
    OUT_OF_REACH = "out_of_reach"
    MAX_ITER = "max_iter"
    SINGULAR = "singular"

    @property
    def is_usable(self) -> bool:
        """Whether the returned configuration is safe to command.

        Every status returns a limit-respecting configuration, so all of them are
        mechanically safe; this flags the ones that actually met the request.
        """
        return self in (IKStatus.CONVERGED, IKStatus.POSITION_ONLY)

    @property
    def rank(self) -> int:
        """Preference order when picking the best of several attempts.

        Defined on the enum so that adding a status cannot silently omit it from
        result ranking.
        """
        return _STATUS_RANK[self]


_STATUS_RANK: dict[IKStatus, int] = {
    IKStatus.CONVERGED: 0,
    IKStatus.POSITION_ONLY: 1,
    IKStatus.LIMIT_BLOCKED: 2,
    IKStatus.OUT_OF_REACH: 3,
    IKStatus.MAX_ITER: 4,
    IKStatus.SINGULAR: 5,
}

# A status added to the enum without a rank is a bug; fail at import time.
assert set(_STATUS_RANK) == set(IKStatus), "IKStatus与_STATUS_RANK不同步"


@dataclass(frozen=True)
class SolverConfig:
    """Tuning for the numerical solvers.

    ``damping_min`` is deliberately non-zero: over a third of this arm's
    configurations have a smallest singular value below 0.01, and an undamped
    pseudo-inverse oscillates or diverges there.
    """

    position_weight: float = 1.0
    orientation_weight: float = 0.35
    position_tolerance: float = 1e-6
    orientation_tolerance: float = 1e-5
    max_iterations: int = 200
    damping_min: float = 1e-4
    damping_init: float = 1e-2
    damping_max: float = 1e2
    damping_decrease: float = 0.4
    damping_increase: float = 2.5
    max_step_norm: float = 0.15
    restarts: int = 24
    seed: int = 0

    def __post_init__(self) -> None:
        if self.position_weight <= 0 or self.orientation_weight < 0:
            raise ValueError("位置权重必须为正，姿态权重不能为负")
        if self.max_iterations < 1 or self.restarts < 1:
            raise ValueError("max_iterations和restarts必须至少为1")
        if not 0 < self.damping_min <= self.damping_init <= self.damping_max:
            raise ValueError("阻尼参数必须满足 0 < min <= init <= max")
        if self.max_step_norm <= 0:
            raise ValueError("max_step_norm必须为正")

    @property
    def position_only(self) -> bool:
        """Whether orientation is ignored entirely."""
        return self.orientation_weight == 0.0

    @property
    def weights(self) -> FloatArray:
        return np.array(
            [self.position_weight] * 3 + [self.orientation_weight] * 3,
            dtype=np.float64,
        )


@dataclass(frozen=True)
class IKResult:
    """Outcome of a solve.

    ``q`` is always within joint limits, even when ``status`` is not
    ``CONVERGED`` -- an unreachable target yields the closest reachable pose
    rather than a diverged or out-of-range configuration.
    """

    status: IKStatus
    q: FloatArray
    position_error: float
    orientation_error: float
    iterations: int
    restarts_used: int = 0
    """Extra seeds tried beyond the first; 0 means the initial seed sufficed."""
    residual_history: tuple[float, ...] = field(default=())

    @property
    def success(self) -> bool:
        return self.status is IKStatus.CONVERGED

    def __str__(self) -> str:
        return (
            f"IKResult({self.status.value}, pos_err={self.position_error * 1000:.3f}mm, "
            f"rot_err={np.degrees(self.orientation_error):.3f}deg, "
            f"iters={self.iterations}, restarts={self.restarts_used})"
        )
