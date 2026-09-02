"""SE(3) and SO(3) helpers.

This module is the single place in the library where rotation representations are
converted. Every other module calls into it so that one convention is used
throughout: URDF's fixed-axis roll-pitch-yaw, meaning ``R = Rz(yaw) @ Ry(pitch)
@ Rx(roll)``.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]

__all__ = [
    "rpy_to_matrix",
    "matrix_to_rpy",
    "make_transform",
    "invert_transform",
    "rotation_log",
    "rotation_exp",
    "axis_angle_to_matrix",
    "pose_error",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
]


def rpy_to_matrix(rpy: npt.ArrayLike) -> FloatArray:
    """Convert URDF roll-pitch-yaw to a rotation matrix."""
    roll, pitch, yaw = (float(v) for v in np.asarray(rpy, dtype=np.float64).reshape(3))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def matrix_to_rpy(matrix: npt.ArrayLike) -> FloatArray:
    """Recover roll-pitch-yaw from a rotation matrix.

    At pitch = +-90 degrees roll and yaw are not separable; roll is set to zero
    and the whole rotation is folded into yaw.
    """
    rot = np.asarray(matrix, dtype=np.float64)[:3, :3]
    sp = -rot[2, 0]
    if abs(sp) >= 1.0 - 1e-12:
        pitch = np.pi / 2 * np.sign(sp)
        return np.array([0.0, pitch, np.arctan2(-rot[0, 1], rot[1, 1])], dtype=np.float64)
    return np.array(
        [
            np.arctan2(rot[2, 1], rot[2, 2]),
            np.arcsin(np.clip(sp, -1.0, 1.0)),
            np.arctan2(rot[1, 0], rot[0, 0]),
        ],
        dtype=np.float64,
    )


def make_transform(
    xyz: npt.ArrayLike = (0.0, 0.0, 0.0),
    rpy: npt.ArrayLike | None = None,
) -> FloatArray:
    """Build a 4x4 homogeneous transform from a translation and optional rpy."""
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = np.asarray(xyz, dtype=np.float64).reshape(3)
    if rpy is not None:
        out[:3, :3] = rpy_to_matrix(rpy)
    return out


def invert_transform(transform: npt.ArrayLike) -> FloatArray:
    """Invert a homogeneous transform without a general matrix inverse."""
    mat = np.asarray(transform, dtype=np.float64)
    rot_t = mat[:3, :3].T
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot_t
    out[:3, 3] = -rot_t @ mat[:3, 3]
    return out


def rotation_log(matrix: npt.ArrayLike) -> FloatArray:
    """SO(3) logarithm: rotation matrix to a rotation vector (axis * angle).

    The returned 3-vector is what the angular part of a geometric Jacobian acts
    on, which is why orientation error is expressed this way rather than as an
    Euler-angle difference.
    """
    rot = np.asarray(matrix, dtype=np.float64)[:3, :3]
    cos_angle = np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-8:
        # Near identity the skew part is already the rotation vector.
        return 0.5 * np.array(
            [rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]],
            dtype=np.float64,
        )
    if np.pi - angle < 1e-6:
        # Near pi the skew part vanishes; recover the axis from R + I instead.
        plus = rot + np.eye(3)
        axis = plus[:, int(np.argmax(np.diag(plus)))]
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return np.zeros(3, dtype=np.float64)
        return (axis / norm) * angle
    factor = angle / (2.0 * np.sin(angle))
    return factor * np.array(
        [rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]],
        dtype=np.float64,
    )


def rotation_exp(rotvec: npt.ArrayLike) -> FloatArray:
    """SO(3) exponential: rotation vector to a rotation matrix (Rodrigues)."""
    vec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vec))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    return axis_angle_to_matrix(vec / angle, angle)


def axis_angle_to_matrix(axis: npt.ArrayLike, angle: float) -> FloatArray:
    """Rodrigues rotation about a unit axis."""
    unit = np.asarray(axis, dtype=np.float64).reshape(3)
    # Check the exact coordinate-axis form before doing a norm/scan.  URDF
    # joints commonly use these axes and this function is called at every FK
    # iteration during IK.
    if abs(unit[0]) == 1.0 and unit[1] == 0.0 and unit[2] == 0.0:
        cosine, sine = np.cos(angle), np.sin(angle) * np.sign(unit[0])
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
            dtype=np.float64,
        )
    if abs(unit[1]) == 1.0 and unit[0] == 0.0 and unit[2] == 0.0:
        cosine, sine = np.cos(angle), np.sin(angle) * np.sign(unit[1])
        return np.array(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
            dtype=np.float64,
        )
    if abs(unit[2]) == 1.0 and unit[0] == 0.0 and unit[1] == 0.0:
        cosine, sine = np.cos(angle), np.sin(angle) * np.sign(unit[2])
        return np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    norm = float(np.linalg.norm(unit))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    unit = unit / norm
    skew = np.array(
        [[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def pose_error(current: npt.ArrayLike, target: npt.ArrayLike) -> FloatArray:
    """Six-vector error from ``current`` to ``target`` as ``[dp, drotvec]``."""
    cur = np.asarray(current, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    out = np.empty(6, dtype=np.float64)
    out[:3] = tgt[:3, 3] - cur[:3, 3]
    out[3:] = rotation_log(tgt[:3, :3] @ cur[:3, :3].T)
    return out


def quaternion_to_matrix(quat: npt.ArrayLike) -> FloatArray:
    """Convert a ``(w, x, y, z)`` quaternion to a rotation matrix."""
    w, x, y, z = (float(v) for v in np.asarray(quat, dtype=np.float64).reshape(4))
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("四元数模长为零")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(matrix: npt.ArrayLike) -> FloatArray:
    """Convert a rotation matrix to a ``(w, x, y, z)`` quaternion."""
    rot = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        return np.array(
            [
                0.25 * scale,
                (rot[2, 1] - rot[1, 2]) / scale,
                (rot[0, 2] - rot[2, 0]) / scale,
                (rot[1, 0] - rot[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    idx = int(np.argmax(np.diag(rot)))
    if idx == 0:
        scale = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        return np.array(
            [
                (rot[2, 1] - rot[1, 2]) / scale,
                0.25 * scale,
                (rot[0, 1] + rot[1, 0]) / scale,
                (rot[0, 2] + rot[2, 0]) / scale,
            ],
            dtype=np.float64,
        )
    if idx == 1:
        scale = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        return np.array(
            [
                (rot[0, 2] - rot[2, 0]) / scale,
                (rot[0, 1] + rot[1, 0]) / scale,
                0.25 * scale,
                (rot[1, 2] + rot[2, 1]) / scale,
            ],
            dtype=np.float64,
        )
    scale = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
    return np.array(
        [
            (rot[1, 0] - rot[0, 1]) / scale,
            (rot[0, 2] + rot[2, 0]) / scale,
            (rot[1, 2] + rot[2, 1]) / scale,
            0.25 * scale,
        ],
        dtype=np.float64,
    )
