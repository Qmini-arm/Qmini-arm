"""Minimal URDF reader for the elements this arm actually uses.

Only link/joint/origin/axis/limit/visual/collision are handled. Writing this
directly against ``xml.etree`` keeps the kinematics core free of heavier URDF
stacks and lets us warn about the specific defects this file has carried, such
as fixed joints that still declare an axis and limits.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .transforms import FloatArray, make_transform

logger = logging.getLogger(__name__)

__all__ = ["UrdfLink", "UrdfJoint", "UrdfGeometry", "UrdfRobot", "parse_urdf"]


@dataclass(frozen=True)
class UrdfGeometry:
    """A visual or collision element resolved to its link frame."""

    origin: FloatArray
    #: Shape type: ``"mesh"`` or ``"box"``. Whether this is a visual or a
    #: collision is already given by which tuple of :class:`UrdfLink` holds it.
    kind: str
    mesh_path: Path | None = None
    mesh_scale: FloatArray | None = None
    box_size: FloatArray | None = None


@dataclass(frozen=True)
class UrdfLink:
    name: str
    visuals: tuple[UrdfGeometry, ...] = ()
    collisions: tuple[UrdfGeometry, ...] = ()
    mass: float = 0.0
    inertia_origin: FloatArray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    #: 3x3 inertia tensor about the inertial origin, or None if not declared.
    inertia: FloatArray | None = None


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: FloatArray
    axis: FloatArray
    lower: float
    upper: float
    effort: float
    velocity: float

    @property
    def is_actuated(self) -> bool:
        return self.joint_type in ("revolute", "continuous", "prismatic")


@dataclass(frozen=True)
class UrdfRobot:
    name: str
    links: dict[str, UrdfLink]
    joints: tuple[UrdfJoint, ...]
    source: Path
    root_link: str


def _floats(text: str | None, count: int, default: tuple[float, ...]) -> FloatArray:
    if text is None:
        return np.array(default, dtype=np.float64)
    tokens = text.split()
    if len(tokens) != count:
        raise ValueError(f"期望{count}个数值，得到{len(tokens)}个：{text!r}")
    try:
        return np.array([float(token) for token in tokens], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"无法解析为浮点数：{text!r}") from exc


def _parse_origin(element: ET.Element | None) -> FloatArray:
    if element is None:
        return np.eye(4, dtype=np.float64)
    node = element.find("origin")
    if node is None:
        return np.eye(4, dtype=np.float64)
    return make_transform(
        _floats(node.get("xyz"), 3, (0.0, 0.0, 0.0)),
        _floats(node.get("rpy"), 3, (0.0, 0.0, 0.0)),
    )


def _parse_geometry(element: ET.Element, base_dir: Path) -> UrdfGeometry | None:
    geometry = element.find("geometry")
    if geometry is None:
        return None
    origin = _parse_origin(element)
    mesh = geometry.find("mesh")
    if mesh is not None:
        filename = mesh.get("filename", "")
        cleaned = filename.split("package://", 1)[-1]
        return UrdfGeometry(
            origin=origin,
            kind="mesh",
            mesh_path=(base_dir / cleaned).resolve(),
            mesh_scale=_floats(mesh.get("scale"), 3, (1.0, 1.0, 1.0)),
        )
    box = geometry.find("box")
    if box is not None:
        return UrdfGeometry(
            origin=origin,
            kind="box",
            box_size=_floats(box.get("size"), 3, (0.0, 0.0, 0.0)),
        )
    logger.warning("跳过不支持的几何类型：link下的%s", list(geometry))
    return None


def _parse_link(element: ET.Element, base_dir: Path) -> UrdfLink:
    name = element.get("name")
    if not name:
        raise ValueError("存在无名link")
    inertial = element.find("inertial")
    mass = 0.0
    inertia: FloatArray | None = None
    if inertial is not None:
        mass_node = inertial.find("mass")
        if mass_node is not None:
            mass = float(mass_node.get("value", "0"))
        node = inertial.find("inertia")
        if node is not None:
            ixx, iyy, izz, ixy, ixz, iyz = (
                float(node.get(key, "0"))
                for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
            )
            inertia = np.array(
                [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64
            )
    return UrdfLink(
        name=name,
        visuals=tuple(
            geom
            for node in element.findall("visual")
            if (geom := _parse_geometry(node, base_dir)) is not None
        ),
        collisions=tuple(
            geom
            for node in element.findall("collision")
            if (geom := _parse_geometry(node, base_dir)) is not None
        ),
        mass=mass,
        inertia_origin=_parse_origin(inertial),
        inertia=inertia,
    )


def _parse_joint(element: ET.Element) -> UrdfJoint:
    name = element.get("name")
    joint_type = element.get("type")
    if not name or not joint_type:
        raise ValueError(f"joint缺少name或type：{element.attrib}")
    parent = element.find("parent")
    child = element.find("child")
    if parent is None or child is None:
        raise ValueError(f"joint {name} 缺少parent或child")
    limit = element.find("limit")
    lower, upper, effort, velocity = -np.pi, np.pi, 0.0, 0.0
    if limit is not None:
        lower = float(limit.get("lower", -np.pi))
        upper = float(limit.get("upper", np.pi))
        effort = float(limit.get("effort", 0.0))
        velocity = float(limit.get("velocity", 0.0))
        if lower > upper:
            raise ValueError(f"joint {name} 的limit下限大于上限")
    if joint_type == "fixed" and (element.find("axis") is not None or limit is not None):
        # Leftover tags from joints demoted to fixed. Harmless — parsers ignore
        # axis/limit on fixed joints — so this is debug rather than a warning, or
        # it would fire seven times on every single load.
        logger.debug(
            "joint %s 是fixed但仍带有axis/limit标签；按固定处理。"
            "若本意是可动关节，请把type改为revolute。",
            name,
        )
    return UrdfJoint(
        name=name,
        joint_type=joint_type,
        parent=str(parent.get("link")),
        child=str(child.get("link")),
        origin=_parse_origin(element),
        axis=_floats(
            None if (axis_node := element.find("axis")) is None else axis_node.get("xyz"),
            3,
            (1.0, 0.0, 0.0),
        ),
        lower=lower,
        upper=upper,
        effort=effort,
        velocity=velocity,
    )


def parse_urdf(path: str | Path) -> UrdfRobot:
    """Parse a URDF and validate that it forms a single-rooted tree."""
    source = Path(path).resolve()
    root = ET.parse(source).getroot()
    base_dir = source.parent

    links: dict[str, UrdfLink] = {}
    for element in root.findall("link"):
        link = _parse_link(element, base_dir)
        if link.name in links:
            raise ValueError(f"link {link.name} 重复定义；请删除多余的一份")
        links[link.name] = link

    joints = tuple(_parse_joint(element) for element in root.findall("joint"))
    names = [joint.name for joint in joints]
    if len(names) != len(set(names)):
        raise ValueError("存在重复的joint名")

    for joint in joints:
        for role, link_name in (("parent", joint.parent), ("child", joint.child)):
            if link_name not in links:
                raise ValueError(f"joint {joint.name} 的{role} link {link_name!r} 未定义")

    children = {joint.child for joint in joints}
    roots = [name for name in links if name not in children]
    if len(roots) != 1:
        raise ValueError(f"URDF必须恰好有一个根link，实际找到{roots}")

    child_counts: dict[str, int] = {}
    for joint in joints:
        child_counts[joint.child] = child_counts.get(joint.child, 0) + 1
    multi = [name for name, count in child_counts.items() if count > 1]
    if multi:
        raise ValueError(f"这些link有多个父关节，不是树结构：{multi}")

    adjacency: dict[str, list[str]] = {}
    for joint in joints:
        adjacency.setdefault(joint.parent, []).append(joint.child)
    seen: set[str] = set()
    stack = [roots[0]]
    while stack:
        current = stack.pop()
        if current in seen:
            raise ValueError(f"检测到环路，link {current} 被重复访问")
        seen.add(current)
        stack.extend(adjacency.get(current, []))
    unreachable = set(links) - seen
    if unreachable:
        raise ValueError(f"这些link从根不可达：{sorted(unreachable)}")

    logger.info(
        "已加载 %s：%d个link，%d个joint(其中%d个可动)",
        source.name,
        len(links),
        len(joints),
        sum(1 for joint in joints if joint.is_actuated),
    )
    return UrdfRobot(
        name=root.get("name", source.stem),
        links=links,
        joints=joints,
        source=source,
        root_link=roots[0],
    )
