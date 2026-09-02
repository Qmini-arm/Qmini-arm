"""Command-line interface: ``arm-ik fk | ik | workspace | viz``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_URDF = "description/arm.urdf"


def _servo_speed(value: str) -> int:
    try:
        speed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("速度必须是整数") from exc
    if not 1 <= speed <= 1023:
        raise argparse.ArgumentTypeError("速度必须在1..1023")
    return speed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arm-ik",
        description="Kinematics and IK for the six-axis CDS55xx arm.",
    )
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="URDF path")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    fk = sub.add_parser("fk", help="forward kinematics")
    fk.add_argument("q", nargs="+", type=float, help="joint angles (radians)")

    ik = sub.add_parser("ik", help="inverse kinematics")
    ik.add_argument("--pos", nargs=3, type=float, required=True),
    ik.add_argument("--rpy", nargs=3, type=float, default=None)
    ik.add_argument("--solver", default="dls", choices=["dls", "least_squares"])
    ik.add_argument("--servo", action="store_true", help="also show servo ticks")

    ws = sub.add_parser("workspace", help="sample/analyse the reachable workspace")
    ws.add_argument("--count", type=int, default=20000)
    ws.add_argument("--seed", type=int, default=0)

    viz = sub.add_parser("viz", help="launch the viser viewer")
    viz.add_argument("--mode", choices=["viewer", "ik", "replay"], default="viewer")
    viz.add_argument("--host", default="0.0.0.0")
    viz.add_argument("--port", type=int, default=8080)
    connection = viz.add_mutually_exclusive_group()
    connection.add_argument(
        "--device",
        "--serial",
        dest="device",
        default=None,
        help="serial port (default: auto-select a unique device)",
    )
    connection.add_argument(
        "--sim",
        action="store_true",
        help="do not open a serial port; keep viewer/ik in simulation mode",
    )
    viz.add_argument(
        "--speed",
        type=_servo_speed,
        default=160,
        help="real-arm goal speed (1..1023) for viewer/ik",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    from arm_ik import RobotModel

    robot = RobotModel.from_urdf(args.urdf)

    if args.command == "fk":
        q = np.array(args.q, dtype=float)
        if q.size != robot.dof:
            print(f"需要{robot.dof}个关节角，得到{q.size}", file=sys.stderr)
            return 2
        pose = robot.fk(q)
        print("position (m):", np.round(pose[:3, 3], 5))
        from arm_ik.model.transforms import matrix_to_rpy

        print("rotation rpy (deg):", np.round(np.degrees(matrix_to_rpy(pose[:3, :3])), 2))
        return 0

    if args.command == "ik":
        result = robot.ik(
            position=args.pos,
            orientation=args.rpy,
            solver=args.solver,
        )
        print(f"status: {result.status.value}")
        print(f"  pos_err: {result.position_error*1000:.3f} mm")
        print(f"  rot_err: {np.degrees(result.orientation_error):.3f} deg")
        print(f"  q (rad): {np.round(result.q, 4)}")
        print(f"  q (deg): {np.round(np.degrees(result.q), 2)}")
        if args.servo:
            from arm_ik.servo import ServoMap

            cal = Path(args.urdf).parent.parent / "arm_ik" / "config" / "servo_calibration.yaml"
            servo_map = ServoMap.from_yaml(cal, robot.joint_names)
            print(f"  servo ticks: {servo_map.to_ticks(result.q)}")
        return 0

    if args.command == "workspace":
        from arm_ik.workspace import analyze_workspace, sample_workspace

        ws = sample_workspace(robot, count=args.count, seed=args.seed)
        rep = analyze_workspace(robot, ws)
        print(f"samples: {rep.samples}")
        print(f"  bounds x: [{rep.bounds_lower[0]:+.3f}, {rep.bounds_upper[0]:+.3f}] m")
        print(f"  bounds y: [{rep.bounds_lower[1]:+.3f}, {rep.bounds_upper[1]:+.3f}] m")
        print(f"  bounds z: [{rep.bounds_lower[2]:+.3f}, {rep.bounds_upper[2]:+.3f}] m")
        print(f"  radius: {rep.min_radius*1000:.1f} .. {rep.max_radius*1000:.1f} mm")
        print(
            f"  cond# median {rep.condition_number_median:.0f}, "
            f"p90 {rep.condition_number_p90:.0f}"
        )
        print(f"  manip median {rep.manipulability_median:.2e}")
        print(f"  wrist cone (95pct): {rep.orientation_cone_half_angle_deg:.1f} deg")
        return 0

    if args.command == "viz":
        from arm_ik.viz import launch_ik_app, launch_viewer, replay

        if args.sim and args.mode == "replay":
            print("--sim只适用于viewer和ik模式", file=sys.stderr)
            return 2

        if args.mode == "viewer":
            if not args.sim:
                from cds_arm import connect

                bus = connect(args.device) if args.device else connect()
                with bus as arm:
                    launch_viewer(
                        robot,
                        servo_backend=arm,
                        speed=args.speed,
                        host=args.host,
                        port=args.port,
                    )
            else:
                launch_viewer(robot, host=args.host, port=args.port)
        elif args.mode == "ik":
            if not args.sim:
                from cds_arm import connect

                bus = connect(args.device) if args.device else connect()
                with bus as arm:
                    launch_ik_app(
                        robot,
                        servo_backend=arm,
                        speed=args.speed,
                        host=args.host,
                        port=args.port,
                    )
            else:
                launch_ik_app(robot, host=args.host, port=args.port)
        else:
            from cds_arm import connect

            bus = connect(args.device) if args.device else connect()
            with bus as arm:
                replay(robot, arm, host=args.host, port=args.port)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
