"""The kinematic model the rest of the library is built on."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..kinematics.forward import ChainState, forward_kinematics, geometric_jacobian
from .transforms import FloatArray, make_transform, quaternion_to_matrix
from .urdf_parser import UrdfJoint, UrdfRobot, parse_urdf

if TYPE_CHECKING:
    from ..config import IKResult

logger = logging.getLogger(__name__)

__all__ = ["RobotModel"]

DEFAULT_TIP = "hand_palm"


@dataclass
class RobotModel:
    """A serial chain with actuated-joint bookkeeping and FK/Jacobian access.

    The degrees of freedom are read from the URDF rather than hard-coded, so
    re-enabling a currently-fixed joint changes the model without code edits.
    """

    urdf: UrdfRobot
    tip_link: str
    joint_names: tuple[str, ...]
    lower: FloatArray
    upper: FloatArray
    velocity: FloatArray
    _actuated_index: dict[str, int]
    _reach_cache: tuple[float, float] | None = None

    @classmethod
    def from_urdf(cls, path: str | Path, tip_link: str = DEFAULT_TIP) -> RobotModel:
        urdf = parse_urdf(path)
        if tip_link not in urdf.links:
            raise ValueError(f"末端link {tip_link!r} 不在URDF中")
        chain = cls._chain_to(urdf, tip_link)
        actuated = [joint for joint in chain if joint.is_actuated]
        if not actuated:
            raise ValueError(f"从{urdf.root_link}到{tip_link}的链上没有可动关节")
        names = tuple(joint.name for joint in actuated)
        model = cls(
            urdf=urdf,
            tip_link=tip_link,
            joint_names=names,
            lower=np.array([joint.lower for joint in actuated], dtype=np.float64),
            upper=np.array([joint.upper for joint in actuated], dtype=np.float64),
            velocity=np.array([joint.velocity for joint in actuated], dtype=np.float64),
            _actuated_index={name: i for i, name in enumerate(names)},
        )
        logger.info("模型 %s：%d自由度，末端=%s", urdf.name, model.dof, tip_link)
        return model

    @staticmethod
    def _chain_to(urdf: UrdfRobot, tip_link: str) -> list[UrdfJoint]:
        """Joints from root to ``tip_link``, in order."""
        by_child = {joint.child: joint for joint in urdf.joints}
        reverse: list[UrdfJoint] = []
        current = tip_link
        while current != urdf.root_link:
            joint = by_child[current]
            reverse.append(joint)
            current = joint.parent
        return list(reversed(reverse))

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def tighten_limits(self, lower: FloatArray, upper: FloatArray) -> None:
        """Narrow the joint limits in place, never widen them.

        Used to fold hardware safety limits into the model so IK only ever
        returns configurations the servo bus will actually execute. Widening is
        rejected because the URDF limits also encode mechanical interference that
        the servo controller knows nothing about.
        """
        lower = np.asarray(lower, dtype=np.float64).reshape(self.dof)
        upper = np.asarray(upper, dtype=np.float64).reshape(self.dof)
        new_lower = np.maximum(self.lower, lower)
        new_upper = np.minimum(self.upper, upper)
        if np.any(new_lower >= new_upper):
            bad = [
                self.joint_names[i]
                for i in np.flatnonzero(new_lower >= new_upper)
            ]
            raise ValueError(f"收紧后限位为空区间: {bad}")
        changed = [
            self.joint_names[i]
            for i in range(self.dof)
            if not (
                np.isclose(new_lower[i], self.lower[i])
                and np.isclose(new_upper[i], self.upper[i])
            )
        ]
        self.lower, self.upper = new_lower, new_upper
        self._reach_cache = None
        if changed:
            logger.info("限位已收紧: %s", ", ".join(changed))

    @property
    def mid_range(self) -> FloatArray:
        """Mid-travel configuration.

        This is the preferred IK seed: the zero configuration of this arm sits at
        a near-singular fully-extended pose and at the edge of one joint's range.
        """
        return 0.5 * (self.lower + self.upper)

    @property
    def reach_bounds(self) -> tuple[float, float]:
        """Min and max tip distance from the base origin, sampled once and cached.

        Used to tell "target is outside the workspace" apart from "the solver
        stalled", which are different problems for the caller.
        """
        if self._reach_cache is None:
            rng = np.random.default_rng(0)
            radii = [
                float(np.linalg.norm(self.fk(self.random_configuration(rng))[:3, 3]))
                for _ in range(2000)
            ]
            radii.append(float(np.linalg.norm(self.fk(np.zeros(self.dof))[:3, 3])))
            # Widen slightly: 2000 samples under-cover the true extremes.
            self._reach_cache = (min(radii) * 0.9, max(radii) * 1.05)
        return self._reach_cache

    def clamp(self, q: npt.ArrayLike) -> FloatArray:
        return np.clip(np.asarray(q, dtype=np.float64).reshape(self.dof), self.lower, self.upper)

    def within_limits(self, q: npt.ArrayLike, tol: float = 1e-9) -> bool:
        """Whether a single config or a ``(N, dof)`` batch all lie within limits."""
        angles = np.asarray(q, dtype=np.float64).reshape(-1, self.dof)
        return bool(
            np.all(angles >= self.lower[None, :] - tol)
            and np.all(angles <= self.upper[None, :] + tol)
        )

    def random_configuration(self, rng: np.random.Generator | None = None) -> FloatArray:
        generator = rng if rng is not None else np.random.default_rng()
        return generator.uniform(self.lower, self.upper)

    def chain_state(self, q: npt.ArrayLike) -> ChainState:
        return forward_kinematics(
            self.urdf.joints, self.urdf.root_link, self.tip_link, q, self._actuated_index
        )

    def fk(self, q: npt.ArrayLike) -> FloatArray:
        """Tip pose as a 4x4 transform in the base frame."""
        return self.chain_state(q).tip_pose

    def link_poses(self, q: npt.ArrayLike) -> dict[str, FloatArray]:
        return self.chain_state(q).link_poses

    def jacobian(self, q: npt.ArrayLike) -> FloatArray:
        return geometric_jacobian(self.chain_state(q))

    def numeric_jacobian(self, q: npt.ArrayLike, eps: float = 1e-7) -> FloatArray:
        """Central-difference Jacobian, used to cross-check the analytic one."""
        from .transforms import rotation_log

        angles = np.asarray(q, dtype=np.float64).reshape(self.dof)
        base = self.fk(angles)
        jac = np.zeros((6, self.dof), dtype=np.float64)
        for i in range(self.dof):
            step = np.zeros(self.dof)
            step[i] = eps
            plus, minus = self.fk(angles + step), self.fk(angles - step)
            jac[:3, i] = (plus[:3, 3] - minus[:3, 3]) / (2 * eps)
            jac[3:, i] = rotation_log(plus[:3, :3] @ base[:3, :3].T) - rotation_log(
                minus[:3, :3] @ base[:3, :3].T
            )
            jac[3:, i] /= 2 * eps
        return jac

    def manipulability(self, q: npt.ArrayLike) -> float:
        """Yoshikawa measure ``sqrt(det(J J^T))``."""
        jac = self.jacobian(q)
        value = float(np.linalg.det(jac @ jac.T))
        return float(np.sqrt(max(value, 0.0)))

    def condition_number(self, q: npt.ArrayLike, length_scale: float = 0.35) -> float:
        """Condition number of the length-scaled Jacobian.

        Raw Jacobians mix metres with radians; dividing the linear rows by a
        characteristic reach makes the number comparable across configurations.
        """
        jac = self.jacobian(q).copy()
        jac[:3, :] /= length_scale
        singular = np.linalg.svd(jac, compute_uv=False)
        if singular[-1] < 1e-15:
            return float("inf")
        return float(singular[0] / singular[-1])

    def ik(
        self,
        position: npt.ArrayLike | None = None,
        orientation: npt.ArrayLike | None = None,
        *,
        target: npt.ArrayLike | None = None,
        seed: npt.ArrayLike | None = None,
        solver: str = "dls",
        config: object | None = None,
    ) -> IKResult:
        """Solve inverse kinematics for a tip pose.

        Pass either ``target`` as a 4x4 transform, or ``position`` plus an
        optional ``orientation`` (rotation matrix, rpy triple, or wxyz
        quaternion). With no orientation the solve is position-only.
        """
        from ..config import SolverConfig
        from ..solvers import get_solver

        if target is None:
            if position is None:
                raise ValueError("必须提供target或position")
            goal = self.target_pose(position, orientation)
        else:
            goal = np.asarray(target, dtype=np.float64)

        cfg = config if config is not None else SolverConfig()
        if orientation is None and target is None:
            cfg = SolverConfig(
                **{**cfg.__dict__, "orientation_weight": 0.0}  # type: ignore[arg-type]
            )
        solver_cls = get_solver(solver)
        return solver_cls(self, cfg).solve(goal, seed)  # type: ignore[arg-type]

    @staticmethod
    def target_pose(
        position: npt.ArrayLike,
        orientation: npt.ArrayLike | None = None,
    ) -> FloatArray:
        """Build a target transform from a position plus rotation matrix, rpy, or quaternion."""
        if orientation is None:
            return make_transform(position)
        rot = np.asarray(orientation, dtype=np.float64)
        if rot.shape == (3, 3):
            out = np.eye(4, dtype=np.float64)
            out[:3, :3] = rot
            out[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
            return out
        if rot.shape == (4,):
            out = np.eye(4, dtype=np.float64)
            out[:3, :3] = quaternion_to_matrix(rot)
            out[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
            return out
        if rot.shape == (3,):
            return make_transform(position, rot)
        raise ValueError(f"无法解释的姿态表示，shape={rot.shape}")
