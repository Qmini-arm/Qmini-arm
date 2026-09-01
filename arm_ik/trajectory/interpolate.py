"""Trajectory interpolation in joint and Cartesian space.

Cartesian orientation is interpolated along the SO(3) geodesic
(``R(t) = R0 exp(t log(R0^T R1))``) rather than linearly in Euler angles, which
would make the wrist rotate at non-constant speed. Every Cartesian waypoint is
solved with IK seeded from the previous point so the joint solution stays on
one branch and does not jump.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..model.transforms import FloatArray, rotation_exp, rotation_log
from .timing import TRAJ_POINTS

if TYPE_CHECKING:
    from ..model.robot_model import RobotModel

__all__ = ["interpolate_joint", "interpolate_cartesian"]


def _quintic_s(t: float) -> float:
    """5th-order ease curve with zero velocity/acceleration at both ends."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def interpolate_joint(
    q0: FloatArray,
    q1: FloatArray,
    points: int = TRAJ_POINTS,
) -> FloatArray:
    """Joint-space path with zero endpoint velocity and acceleration."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    ts = np.linspace(0.0, 1.0, points)
    s = np.array([_quintic_s(t) for t in ts])
    return q0[None, :] * (1 - s)[:, None] + q1[None, :] * s[:, None]


def interpolate_cartesian(
    robot: RobotModel,
    pose0: FloatArray,
    pose1: FloatArray,
    points: int = TRAJ_POINTS,
) -> tuple[FloatArray, FloatArray]:
    """Cartesian path from ``pose0`` to ``pose1`` with per-point IK.

    Returns ``(q_path, pose_path)`` where ``q_path`` are the IK-solved joint
    vectors. Orientation follows the SO(3) geodesic; position follows a straight
    line in Cartesian space.
    """
    p0, p1 = pose0[:3, 3], pose1[:3, 3]
    r0, r1 = pose0[:3, :3], pose1[:3, :3]
    delta = rotation_log(r0.T @ r1)  # fixed rotation vector in the end frame

    ts = np.linspace(0.0, 1.0, points)
    q_seed = None
    q_path = np.empty((points, robot.dof), dtype=float)
    pose_path = np.empty((points, 4, 4), dtype=float)

    for i, t in enumerate(ts):
        s = _quintic_s(t)
        pt = p0 * (1 - s) + p1 * s
        rot = r0 @ rotation_exp(delta * s)
        target = np.eye(4)
        target[:3, :3] = rot
        target[:3, 3] = pt
        result = robot.ik(target=target, seed=q_seed)
        if not result.status.is_usable:
            raise ValueError(
                f"第{i}个路点(占比{t:.2f})无法求解IK：{result.status.value}，"
                f"pos_err={result.position_error:.4f}m。"
                f"通常是该中间姿态超出可达范围——窄行程臂的直线笛卡尔路径"
                f"常经过臂够不到的地方。try a joint-space path or break the"
                f" Cartesian path into reachable segments."
            )
        q_seed = result.q
        q_path[i] = result.q
        pose_path[i] = robot.fk(result.q)
    return q_path, pose_path
