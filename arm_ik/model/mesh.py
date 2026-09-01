"""Minimal binary/ASCII STL reader.

Only vertex positions are needed here, for two purposes: deriving collision
boxes and inertia from the actual geometry, and feeding the viser viewer. A
full mesh library would work too, but a reader this small avoids the
trimesh/lxml dependency chain for what amounts to one struct unpack.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .transforms import FloatArray

logger = logging.getLogger(__name__)

_BINARY_HEADER = 84
_BYTES_PER_TRIANGLE = 50


@dataclass(frozen=True)
class TriangleMesh:
    """Triangle soup in the mesh's own frame, in metres after scaling."""

    vertices: FloatArray  # (n, 3)
    faces: npt.NDArray[np.int32]  # (m, 3) indices into vertices

    @property
    def aabb(self) -> tuple[FloatArray, FloatArray]:
        """Axis-aligned bounds as ``(lower, upper)``."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def extent(self) -> FloatArray:
        lower, upper = self.aabb
        return upper - lower

    @property
    def aabb_centre(self) -> FloatArray:
        lower, upper = self.aabb
        return (lower + upper) / 2.0

    def transformed(self, transform: FloatArray) -> TriangleMesh:
        """Return a copy with a 4x4 homogeneous ``transform`` applied."""
        rotated = self.vertices @ transform[:3, :3].T + transform[:3, 3]
        return TriangleMesh(vertices=rotated, faces=self.faces)


def _is_binary(path: Path) -> bool:
    """Binary STL declares its triangle count; check it against the file size."""
    size = path.stat().st_size
    if size < _BINARY_HEADER:
        return False
    with path.open("rb") as handle:
        handle.seek(80)
        count = struct.unpack("<I", handle.read(4))[0]
    return size == _BINARY_HEADER + count * _BYTES_PER_TRIANGLE


def _load_binary(path: Path) -> FloatArray:
    raw = np.fromfile(path, dtype=np.uint8, offset=_BINARY_HEADER)
    count = raw.size // _BYTES_PER_TRIANGLE
    if count == 0:
        raise ValueError(f"{path.name} 不含三角面")
    block = raw[: count * _BYTES_PER_TRIANGLE].reshape(count, _BYTES_PER_TRIANGLE)
    # Each 50-byte record is: normal (12B), three vertices (36B), attribute (2B).
    corners = block[:, 12:48].copy().view(np.float32).reshape(count, 3, 3)
    return corners.astype(np.float64)


def _load_ascii(path: Path) -> FloatArray:
    values: list[tuple[float, float, float]] = []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "vertex":
            values.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not values or len(values) % 3:
        raise ValueError(f"{path.name} 的ASCII STL顶点数不是3的倍数")
    return np.asarray(values, dtype=np.float64).reshape(-1, 3, 3)


def load_stl(path: Path, scale: FloatArray | tuple[float, float, float]) -> TriangleMesh:
    """Load ``path`` and apply a per-axis ``scale`` (URDF meshes are in mm)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"网格文件不存在: {path}")
    corners = _load_binary(path) if _is_binary(path) else _load_ascii(path)
    corners = corners * np.asarray(scale, dtype=np.float64)
    flat = corners.reshape(-1, 3)
    # STL has no shared-vertex table; dedupe so the viewer gets a compact mesh.
    vertices, inverse = np.unique(flat, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int32)
    logger.debug("%s: %d三角面 -> %d顶点", path.name, len(faces), len(vertices))
    return TriangleMesh(vertices=vertices, faces=faces)


def box_inertia(mass: float, extent: FloatArray) -> FloatArray:
    """Inertia tensor of a solid box of ``extent`` about its own centroid.

    Returns the diagonal ``(ixx, iyy, izz)``; the off-diagonal terms of a
    box aligned with its principal axes are zero.
    """
    x, y, z = np.asarray(extent, dtype=np.float64)
    factor = mass / 12.0
    return np.array(
        [factor * (y * y + z * z), factor * (x * x + z * z), factor * (x * x + y * y)],
        dtype=np.float64,
    )
