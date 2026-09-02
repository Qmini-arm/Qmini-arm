"""Forward kinematics and the analytic geometric Jacobian.

Both are computed from a single chain walk: the Jacobian needs each joint's axis
direction and origin in the base frame, which the FK pass already produces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..model.transforms import FloatArray, axis_angle_to_matrix

__all__ = ["ChainState", "forward_kinematics", "geometric_jacobian"]


@dataclass(frozen=True)
class ChainState:
    """Everything one FK pass yields.

    ``link_poses`` feeds collision checking and visualisation; ``axis_dirs`` and
    ``axis_origins`` feed the Jacobian.
    """

    link_poses: dict[str, FloatArray]
    axis_dirs: FloatArray
    axis_origins: FloatArray
    tip_pose: FloatArray


def forward_kinematics(
    joints: tuple,
    root_link: str,
    tip_link: str,
    q: npt.ArrayLike,
    actuated_index: dict[str, int],
) -> ChainState:
    """Walk the tree once, accumulating every link pose in the base frame.

    ``joints`` is the full joint tuple in document order; fixed joints simply
    contribute their static origin transform.
    """
    angles = np.asarray(q, dtype=np.float64).reshape(-1)
    poses: dict[str, FloatArray] = {root_link: np.eye(4, dtype=np.float64)}
    dof = len(actuated_index)
    axis_dirs = np.zeros((dof, 3), dtype=np.float64)
    axis_origins = np.zeros((dof, 3), dtype=np.float64)

    for joint in joints:
        parent_pose = poses.get(joint.parent)
        if parent_pose is None:
            # Not yet reachable in document order; the parser guarantees the
            # tree is well formed, so a second pass would resolve it. This arm
            # is declared parent-before-child throughout.
            continue
        local = joint.origin
        index = actuated_index.get(joint.name)
        if index is not None:
            rot = axis_angle_to_matrix(joint.axis, float(angles[index]))
            motion = np.eye(4, dtype=np.float64)
            motion[:3, :3] = rot
            local = local @ motion
            joint_pose = parent_pose @ joint.origin
            axis_dirs[index] = joint_pose[:3, :3] @ joint.axis
            axis_origins[index] = joint_pose[:3, 3]
        poses[joint.child] = parent_pose @ local

    return ChainState(
        link_poses=poses,
        axis_dirs=axis_dirs,
        axis_origins=axis_origins,
        tip_pose=poses[tip_link],
    )


def geometric_jacobian(state: ChainState) -> FloatArray:
    """Build the 6xN geometric Jacobian from a completed FK pass.

    Column i of a revolute joint is ``[z_i x (p_tip - p_i); z_i]``.
    """
    tip = state.tip_pose[:3, 3]
    dof = state.axis_dirs.shape[0]
    jac = np.empty((6, dof), dtype=np.float64)
    # Expand the cross product component-wise.  For this small fixed-size
    # Jacobian it avoids NumPy's general cross-product dispatch overhead.
    axis = state.axis_dirs
    delta = tip - state.axis_origins
    jac[0] = axis[:, 1] * delta[:, 2] - axis[:, 2] * delta[:, 1]
    jac[1] = axis[:, 2] * delta[:, 0] - axis[:, 0] * delta[:, 2]
    jac[2] = axis[:, 0] * delta[:, 1] - axis[:, 1] * delta[:, 0]
    jac[3:] = state.axis_dirs.T
    return jac
