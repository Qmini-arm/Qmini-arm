"""Exact box-box self-collision checking via the separating axis theorem.

Every collision element in ``arm.urdf`` is a ``<box>``, so oriented-bounding-box
overlap is exact here rather than a conservative approximation. Two OBBs are
disjoint if and only if one of 15 candidate axes separates them: the 3 face
normals of each box plus the 9 pairwise cross products of their edge directions.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..model.transforms import FloatArray

logger = logging.getLogger(__name__)


class UrdfLike(Protocol):
    links: dict
    joints: Sequence


class RobotLike(Protocol):
    """The part of :class:`~arm_ik.RobotModel` this module needs."""

    urdf: UrdfLike

    def link_poses(self, q: FloatArray) -> dict[str, FloatArray]: ...


@dataclass(frozen=True)
class CollisionPair:
    """Two link names found to be in collision, with penetration depth."""

    link_a: str
    link_b: str
    depth: float

    def __str__(self) -> str:
        return f"{self.link_a}<->{self.link_b} ({self.depth * 1000:.1f}mm)"


def _obb_overlap(
    center_a: FloatArray,
    rot_a: FloatArray,
    half_a: FloatArray,
    center_b: FloatArray,
    rot_b: FloatArray,
    half_b: FloatArray,
) -> float:
    """Return penetration depth along the least-overlapping axis, 0.0 if disjoint.

    Args:
        center_a: World-frame centre of box A.
        rot_a: World-frame rotation of box A; columns are its edge directions.
        half_a: Half-extents of box A along its own axes.
        center_b: World-frame centre of box B.
        rot_b: World-frame rotation of box B.
        half_b: Half-extents of box B along its own axes.

    Returns:
        The smallest overlap across all separating-axis candidates, or ``0.0``
        as soon as any axis separates the two boxes.
    """
    delta = center_b - center_a
    axes: list[FloatArray] = [rot_a[:, i] for i in range(3)]
    axes += [rot_b[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            axis = np.cross(rot_a[:, i], rot_b[:, j])
            norm = float(np.linalg.norm(axis))
            # Near-parallel edge pairs give a degenerate cross product; the face
            # normals already cover those configurations.
            if norm > 1e-9:
                axes.append(axis / norm)

    min_overlap = np.inf
    for axis in axes:
        reach_a = float(np.abs(rot_a.T @ axis) @ half_a)
        reach_b = float(np.abs(rot_b.T @ axis) @ half_b)
        distance = abs(float(delta @ axis))
        overlap = reach_a + reach_b - distance
        if overlap <= 0.0:
            return 0.0
        min_overlap = min(min_overlap, overlap)
    return float(min_overlap)


class CollisionChecker:
    """Check a configuration for link-against-link overlap.

    Adjacent links always touch by construction, so pairs joined by a joint are
    exempt. Pairs separated by a single intermediate link are also exempt: the
    URDF's box approximations of neighbouring brackets overlap slightly even in
    valid poses.
    """

    def __init__(
        self,
        robot: RobotLike,
        extra_ignored: Iterable[tuple[str, str]] = (),
        margin: float = 0.0,
    ) -> None:
        self.robot = robot
        self.margin = margin
        # A link may declare several collision geometries, so key by link name and
        # keep every box. Non-box geometry is skipped: this URDF uses boxes
        # throughout, and an exact OBB test is only valid for boxes.
        self.boxes: dict[str, list[tuple[FloatArray, FloatArray]]] = {}
        self.radii: dict[str, list[float]] = {}
        for name, link in robot.urdf.links.items():
            geoms = [
                (geom.origin, geom.box_size / 2.0)
                for geom in link.collisions
                if geom.kind == "box" and geom.box_size is not None
            ]
            if geoms:
                self.boxes[name] = geoms
                # Bounding-sphere radius per box, for broad-phase culling. The
                # margin is folded in so culling never discards a pair the
                # narrow-phase test would have flagged.
                self.radii[name] = [
                    float(np.linalg.norm(half)) + margin for _, half in geoms
                ]
            skipped = len(link.collisions) - len(geoms)
            if skipped:
                logger.warning(
                    "link %s 有%d个非box碰撞体，自碰撞检测将忽略它们", name, skipped
                )
        self.ignored = self._build_exemptions(robot)
        self.ignored |= {frozenset(pair) for pair in extra_ignored}
        self.pairs = [
            (a, b)
            for a, b in itertools.combinations(sorted(self.boxes), 2)
            if frozenset((a, b)) not in self.ignored
        ]
        logger.debug(
            "自碰撞检测: %d个盒体, %d对豁免, %d对待检",
            len(self.boxes),
            len(self.ignored),
            len(self.pairs),
        )

    @staticmethod
    def _build_exemptions(robot: RobotLike) -> set[frozenset[str]]:
        """Exempt links within two hops of each other on the kinematic tree."""
        neighbours: dict[str, set[str]] = {name: set() for name in robot.urdf.links}
        for joint in robot.urdf.joints:
            neighbours[joint.parent].add(joint.child)
            neighbours[joint.child].add(joint.parent)
        exempt: set[frozenset[str]] = set()
        for name, direct in neighbours.items():
            for other in direct:
                exempt.add(frozenset((name, other)))
                for second in neighbours[other]:
                    if second != name:
                        exempt.add(frozenset((name, second)))
        return exempt

    def check(self, q: FloatArray) -> list[CollisionPair]:
        """Return every colliding link pair in configuration ``q``."""
        poses = self.robot.link_poses(q)
        world: dict[str, list[FloatArray]] = {
            name: [poses[name] @ origin for origin, _ in boxes]
            for name, boxes in self.boxes.items()
        }
        found: list[CollisionPair] = []
        for name_a, name_b in self.pairs:
            deepest = 0.0
            for index_a, (_, half_a) in enumerate(self.boxes[name_a]):
                world_a = world[name_a][index_a]
                radius_a = self.radii[name_a][index_a]
                for index_b, (_, half_b) in enumerate(self.boxes[name_b]):
                    world_b = world[name_b][index_b]
                    gap = float(np.linalg.norm(world_a[:3, 3] - world_b[:3, 3]))
                    if gap > radius_a + self.radii[name_b][index_b]:
                        continue
                    depth = _obb_overlap(
                        world_a[:3, 3],
                        world_a[:3, :3],
                        half_a + self.margin,
                        world_b[:3, 3],
                        world_b[:3, :3],
                        half_b + self.margin,
                    )
                    deepest = max(deepest, depth)
            if deepest > 0.0:
                found.append(CollisionPair(name_a, name_b, deepest))
        return found

    def is_free(self, q: FloatArray) -> bool:
        """Whether ``q`` is collision-free."""
        return not self.check(q)
