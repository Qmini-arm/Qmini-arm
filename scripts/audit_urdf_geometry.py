"""Compare each link's collision box and inertia against its visual mesh.

The URDF's visual origins were edited by hand; the collision boxes and inertia
tensors were not updated to match. This script measures the actual STL bounds
in the visual frame and reports the corrected values, so the fix is derived
from geometry rather than guessed.

Run read-only:      python scripts/audit_urdf_geometry.py
Emit a patched file: python scripts/audit_urdf_geometry.py --write out.urdf
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np

from arm_ik.model.mesh import box_inertia, load_stl
from arm_ik.model.urdf_parser import parse_urdf

logger = logging.getLogger(__name__)

TOLERANCE_M = 1e-4


def _rotated_extent(mesh, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return AABB centre and extent of ``mesh`` after applying ``origin``."""
    placed = mesh.transformed(origin)
    return placed.aabb_centre, placed.extent


def audit(urdf_path: Path) -> list[dict]:
    robot = parse_urdf(urdf_path)
    rows: list[dict] = []
    for name, link in robot.links.items():
        meshes = [g for g in link.visuals if g.kind == "mesh" and g.mesh_path]
        if not meshes:
            continue
        centres, extents = [], []
        for geom in meshes:
            mesh = load_stl(geom.mesh_path, geom.mesh_scale)
            centre, extent = _rotated_extent(mesh, geom.origin)
            centres.append(centre)
            extents.append(extent)
        # Union AABB across every visual mesh on this link.
        pairs = list(zip(centres, extents, strict=True))
        lower = np.min([c - e / 2 for c, e in pairs], axis=0)
        upper = np.max([c + e / 2 for c, e in pairs], axis=0)
        want_centre, want_extent = (lower + upper) / 2, upper - lower

        boxes = [g for g in link.collisions if g.kind == "box" and g.box_size is not None]
        row = {
            "link": name,
            "want_centre": want_centre,
            "want_extent": want_extent,
            "mass": link.mass,
        }
        if boxes:
            row["has_centre"] = boxes[0].origin[:3, 3]
            row["has_extent"] = np.asarray(boxes[0].box_size, dtype=float)
            row["centre_err"] = float(np.abs(row["has_centre"] - want_centre).max())
            row["extent_err"] = float(np.abs(row["has_extent"] - want_extent).max())
        if link.mass and link.inertia is not None:
            row["has_inertia"] = np.diag(link.inertia).copy()
            row["want_inertia"] = box_inertia(link.mass, want_extent)
            row["com_offset"] = float(
                np.abs(link.inertia_origin[:3, 3] - want_centre).max()
            )
        rows.append(row)
    return rows


def _report(rows: list[dict]) -> None:
    bad_box = [r for r in rows if r.get("centre_err", 0) > TOLERANCE_M
               or r.get("extent_err", 0) > TOLERANCE_M]
    print(f"{'link':<16} {'centre err':>11} {'extent err':>11}  {'current size':>22} -> measured")
    print("-" * 96)
    for row in rows:
        if "has_extent" not in row:
            print(f"{row['link']:<16} {'no box':>11}")
            continue
        flag = "  <-- fix" if row in bad_box else ""
        print(
            f"{row['link']:<16} "
            f"{row['centre_err'] * 1e3:>8.1f} mm {row['extent_err'] * 1e3:>8.1f} mm  "
            f"{np.array2string(row['has_extent'] * 1e3, precision=1, floatmode='fixed'):>22} -> "
            f"{np.array2string(row['want_extent'] * 1e3, precision=1, floatmode='fixed')}{flag}"
        )
    print(f"\n{len(bad_box)} / {len(rows)} links need a collision-box fix")

    print(f"\n{'link':<16} {'inertia ratio (current/measured)':<38} CoM offset")
    print("-" * 72)
    for row in rows:
        if "has_inertia" not in row:
            continue
        ratio = row["has_inertia"] / np.where(row["want_inertia"] > 0, row["want_inertia"], 1)
        print(
            f"{row['link']:<16} "
            f"{np.array2string(ratio, precision=2, floatmode='fixed'):<38} "
            f"{row['com_offset'] * 1e3:>6.1f} mm"
        )


def _fmt(values: np.ndarray) -> str:
    """Format a vector the way the rest of the URDF writes numbers."""
    return " ".join(f"{v:g}" for v in np.round(values, 6) + 0.0)


def _sub_attr(text: str, attr: str, value: str) -> str:
    """Replace ``attr="..."`` in a single tag, keeping surrounding formatting."""
    return re.sub(rf'{attr}="[^"]*"', f'{attr}="{value}"', text, count=1)


def _patch_block(block: str, row: dict) -> tuple[str, int]:
    """Correct the collision boxes and inertia inside one ``<link>`` block.

    Operates on raw text rather than a parsed tree so that comments, blank
    lines, and attribute spacing elsewhere in the file survive untouched --
    a reformatted URDF makes the review diff useless.
    """
    centre, extent = _fmt(row["want_centre"]), _fmt(row["want_extent"])
    changed = 0

    def fix_collision(match: re.Match[str]) -> str:
        nonlocal changed
        body = match.group(0)
        if "<box" not in body:
            return body
        changed += 1
        body = _sub_attr(body, "size", extent)
        if "<origin" in body:
            return _sub_attr(body, "xyz", centre)
        indent = match.group(1)
        return body.replace(
            f"{indent}<collision>",
            f'{indent}<collision>\n{indent}  <origin xyz="{centre}"/>',
            1,
        )

    block = re.sub(r"([ \t]*)<collision>.*?</collision>", fix_collision, block, flags=re.S)

    if "want_inertia" in row:
        ixx, iyy, izz = row["want_inertia"]

        def fix_inertial(match: re.Match[str]) -> str:
            body = match.group(0)
            body = _sub_attr(body, "xyz", centre)
            for attr, value in (
                ("ixx", f"{ixx:.9g}"), ("iyy", f"{iyy:.9g}"), ("izz", f"{izz:.9g}"),
                ("ixy", "0"), ("ixz", "0"), ("iyz", "0"),
            ):
                body = _sub_attr(body, attr, value)
            return body

        block = re.sub(r"[ \t]*<inertial>.*?</inertial>", fix_inertial, block, flags=re.S)
    return block, changed


def patch(urdf_path: Path, out_path: Path, rows: list[dict]) -> int:
    """Write ``urdf_path`` to ``out_path`` with collision/inertia corrected.

    Declared masses are kept as-is: they are the user's estimates and cannot be
    recovered from geometry. Only the inertia tensor and the centroid location
    are made consistent with the measured extent.
    """
    text = urdf_path.read_text()
    by_name = {row["link"]: row for row in rows}
    total = 0

    def fix_link(match: re.Match[str]) -> str:
        nonlocal total
        row = by_name.get(match.group(1))
        if row is None:
            return match.group(0)
        patched, count = _patch_block(match.group(0), row)
        total += count
        return patched

    text = re.sub(r'<link name="([^"]+)">.*?</link>', fix_link, text, flags=re.S)
    out_path.write_text(text)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=Path("description/arm.urdf"))
    parser.add_argument("--write", type=Path, help="写出修正后的URDF到该路径")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    rows = audit(args.urdf)
    _report(rows)
    if args.write:
        count = patch(args.urdf, args.write, rows)
        print(f"\n已修正 {count} 个collision几何 -> {args.write}")
        print("注意: mass 保持原值，需实测称重后更新")


if __name__ == "__main__":
    main()
