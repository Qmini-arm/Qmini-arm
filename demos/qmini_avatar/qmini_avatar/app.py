#!/usr/bin/env python3
"""Qmini Avatar: local-camera hand, wrist, and arm teleoperation.

The program is deliberately paused at startup.  ``C`` captures a relative
neutral pose and ``SPACE`` explicitly starts/stops command submission.  With no
serial ports supplied it is a camera + kinematics simulation and never opens a
robot connection.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
from qmini_avatar.core import (
    AvatarMapper,
    AvatarMappingConfig,
    AvatarMotionWorker,
    AvatarPlanner,
    AvatarTarget,
    FingerCommandFilter,
    FingerVisionCalibration,
    HumanHandPose,
    ReachableWorkspaceProjector,
    SerialPortInfo,
    TargetSmoother,
    choose_avatar_serial_ports,
    extract_hand_pose,
    score_avatar_serial_ports,
)

from arm_ik import RobotModel
from arm_ik.collision import CollisionChecker
from arm_ik.servo import ServoMap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_URDF = REPOSITORY_ROOT / "description" / "arm.urdf"
DEFAULT_CALIBRATION = REPOSITORY_ROOT / "arm_ik" / "config" / "servo_calibration.yaml"
DEFAULT_FINGER_CALIBRATION = PROJECT_ROOT / "config" / "finger_calibration.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local MediaPipe teleoperation for Qmini arm + uHand",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument(
        "--live",
        action="store_true",
        help="auto-detect and connect both the Qmini arm and uHand",
    )
    parser.add_argument(
        "--list-ports", action="store_true", help="show arm/hand role scores"
    )
    parser.add_argument(
        "--arm-port", default=None, help="e.g. /dev/cu.usbserial-* or auto"
    )
    parser.add_argument(
        "--hand-port", default=None, help="e.g. /dev/cu.usbmodem* or auto"
    )
    parser.add_argument("--arm-speed", type=int, default=120)
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--hand-hz", type=float, default=30.0)
    parser.add_argument("--loss-timeout", type=float, default=0.45)
    parser.add_argument("--max-joint-step-deg", type=float, default=10.0)
    parser.add_argument("--collision-samples", type=int, default=10)
    parser.add_argument("--workspace-samples", type=int, default=6000)
    parser.add_argument("--workspace-tolerance-mm", type=float, default=12.0)
    parser.add_argument("--smoothing", type=float, default=0.45)
    parser.add_argument(
        "--finger-calibration",
        type=Path,
        default=DEFAULT_FINGER_CALIBRATION,
        help="per-finger visual calibration JSON",
    )
    parser.add_argument("--finger-smoothing", type=float, default=0.65)
    parser.add_argument("--finger-max-step", type=float, default=12.0)
    parser.add_argument("--finger-deadband", type=float, default=0.5)
    parser.add_argument("--finger-keepalive", type=float, default=0.25)
    parser.add_argument("--depth-gain", type=float, default=0.14)
    parser.add_argument("--lateral-gain", type=float, default=0.25)
    parser.add_argument("--vertical-gain", type=float, default=0.25)
    parser.add_argument("--roll-gain", type=float, default=1.0)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--servo-calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--self-test", action="store_true", help="test mapping/IK without camera"
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0:
        raise ValueError("camera width and height must be positive")
    if not 1 <= args.arm_speed <= 1023:
        raise ValueError("arm-speed must be in 1..1023")
    if args.control_hz <= 0 or args.hand_hz <= 0:
        raise ValueError("control frequencies must be positive")
    if args.loss_timeout <= 0:
        raise ValueError("loss-timeout must be positive")
    if args.max_joint_step_deg <= 0 or args.collision_samples < 2:
        raise ValueError("planner safety settings are invalid")
    if args.workspace_samples < 100 or args.workspace_tolerance_mm <= 0:
        raise ValueError("workspace sampling settings are invalid")
    if not 0 < args.smoothing <= 1:
        raise ValueError("smoothing must be in (0, 1]")
    if not 0 < args.finger_smoothing <= 1:
        raise ValueError("finger-smoothing must be in (0, 1]")
    if (
        args.finger_max_step <= 0
        or args.finger_deadband < 0
        or args.finger_keepalive <= 0
    ):
        raise ValueError("finger output settings are invalid")


def load_stack(
    args: argparse.Namespace,
) -> tuple[RobotModel, ServoMap, AvatarPlanner, ReachableWorkspaceProjector]:
    robot = RobotModel.from_urdf(args.urdf)
    servo_map = ServoMap.from_yaml(args.servo_calibration, robot.joint_names)
    for problem in servo_map.validate_against(robot):
        # The measured zero ticks (axis 1 = 812, axis 5 = 359) are authoritative.
        # Show the remaining URDF-vs-safe-window mismatch once per joint, not the
        # derived residual's generic zero/direction warning as a second alarm.
        if "URDF允许" in problem:
            print(f"Joint-limit note: {problem}")
    robot.tighten_limits(*servo_map.effective_limits(robot))
    checker = CollisionChecker(robot)
    print(
        f"Sampling {args.workspace_samples} joint configurations for reachable space..."
    )
    started = time.monotonic()
    workspace = ReachableWorkspaceProjector.sample(
        robot,
        checker,
        count=args.workspace_samples,
        coverage_tolerance_m=args.workspace_tolerance_mm / 1000.0,
    )
    lower, upper = workspace.bounds
    print(
        f"Reachable workspace ready: {workspace.size} collision-free samples "
        f"in {time.monotonic() - started:.1f}s; "
        f"XYZ mm {np.round(lower * 1000).astype(int).tolist()} .. "
        f"{np.round(upper * 1000).astype(int).tolist()}"
    )
    planner = AvatarPlanner(
        robot,
        servo_map,
        collision_checker=checker,
        workspace=workspace,
        max_joint_step_deg=args.max_joint_step_deg,
        collision_samples=args.collision_samples,
    )
    return robot, servo_map, planner, workspace


def discover_serial_ports() -> list[SerialPortInfo]:
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError("pyserial is required for hardware discovery") from exc
    return [
        SerialPortInfo(
            device=str(getattr(item, "device", "")),
            description=str(getattr(item, "description", "")),
            hwid=str(getattr(item, "hwid", "")),
        )
        for item in list_ports.comports()
        if str(getattr(item, "device", ""))
    ]


def print_serial_ports() -> None:
    scores = score_avatar_serial_ports(discover_serial_ports())
    if not scores:
        print("No serial ports found")
        return
    print("Serial devices (higher score means a stronger role match):")
    for item in scores:
        print(
            f"  {item.port.device}: arm={item.arm:3d} hand={item.hand:3d}  "
            f"{item.port.description}  {item.port.hwid}"
        )


def resolve_hardware_ports(args: argparse.Namespace) -> tuple[str | None, str | None]:
    want_arm = bool(args.live or args.arm_port is not None)
    want_hand = bool(args.live or args.hand_port is not None)
    if not want_arm and not want_hand:
        return None, None
    ports = discover_serial_ports()
    arm_override = args.arm_port if args.arm_port is not None else "auto"
    hand_override = args.hand_port if args.hand_port is not None else "auto"
    return choose_avatar_serial_ports(
        ports,
        want_arm=want_arm,
        want_hand=want_hand,
        arm_override=arm_override,
        hand_override=hand_override,
    )


def run_self_test(args: argparse.Namespace) -> int:
    robot, _, planner, _ = load_stack(args)
    finger_calibration = FingerVisionCalibration.load(args.finger_calibration)
    q0 = robot.mid_range
    pose0 = robot.fk(q0)
    human = HumanHandPose(
        palm_x=0.5,
        palm_y=0.5,
        palm_scale=0.16,
        roll=0.0,
        closures=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    mapper = AvatarMapper(robot.servo6_limits)
    mapper.calibrate(human, pose0[:3, 3], q0[robot.servo6_index])
    target = mapper.map(human)
    planned = planner.plan(q0, target)
    assert np.allclose(target.position, pose0[:3, 3])
    assert planned.position_error < 1e-5
    assert finger_calibration.angles_for_closures((0.0,) * 5) == (180,) * 5
    assert finger_calibration.angles_for_closures((1.0,) * 5) == (0,) * 5
    print("Qmini Avatar self-test passed")
    print(f"  neutral position: {np.round(target.position * 1000, 1).tolist()} mm")
    print(f"  IK error: {planned.position_error * 1000:.4f} mm")
    print(f"  finger calibration: {args.finger_calibration}")
    print("  no camera or serial port was opened")
    return 0


def _put_lines(
    cv2: Any, frame: Any, lines: list[tuple[str, tuple[int, int, int]]]
) -> None:
    y = 26
    for text, color in lines:
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 25


def _mapping_config(args: argparse.Namespace) -> AvatarMappingConfig:
    return AvatarMappingConfig(
        depth_gain_m=args.depth_gain,
        lateral_gain_m=args.lateral_gain,
        vertical_gain_m=args.vertical_gain,
        roll_gain=args.roll_gain,
    )


def run_camera(args: argparse.Namespace) -> int:
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        print(
            "Missing camera dependencies. Install demos/requirements.txt in a virtualenv.",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 2

    robot, servo_map, planner, _ = load_stack(args)
    arm_port, hand_port = resolve_hardware_ports(args)
    if arm_port is not None or hand_port is not None:
        print(
            f"Detected hardware: arm={arm_port or 'simulation'}, hand={hand_port or 'simulation'}"
        )
    capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        capture.release()
        return 2

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    drawing = mp.solutions.drawing_utils
    mapper = AvatarMapper(robot.servo6_limits, _mapping_config(args))
    smoother = TargetSmoother(alpha=args.smoothing)
    finger_calibration = FingerVisionCalibration.load(args.finger_calibration)
    finger_filter = FingerCommandFilter(
        finger_calibration,
        alpha=args.finger_smoothing,
        max_step_deg=args.finger_max_step,
    )
    print(f"Finger mapping: calibrated from {args.finger_calibration}")

    with ExitStack() as stack:
        arm = None
        hand = None
        initial_q = robot.mid_range.copy()
        try:
            if arm_port is not None:
                from cds_arm import connect as connect_arm

                print(
                    "Holding the arm at its current pose; keep a hardware power cut-off ready."
                )
                arm = stack.enter_context(connect_arm(arm_port))
                initial_ticks = arm.takeover_current(speed=args.arm_speed)
                initial_q = servo_map.to_joints(initial_ticks)
            if hand_port is not None:
                from uhand import connect as connect_hand

                hand = stack.enter_context(connect_hand(hand_port))
        except Exception:
            capture.release()
            hands.close()
            raise

        motion = AvatarMotionWorker(
            planner,
            initial_q,
            backend=arm,
            speed=args.arm_speed,
            rate_hz=args.control_hz,
        )
        stack.callback(motion.close)

        active = False
        latest_pose: HumanHandPose | None = None
        latest_raw_target: AvatarTarget | None = None
        last_seen = 0.0
        last_hand_update = 0.0
        last_hand_send = 0.0
        last_sent_angles: tuple[int, ...] | None = None
        hand_interval = 1.0 / args.hand_hz
        notice = "Show one hand, press C to calibrate, then SPACE to start"
        fps = 0.0
        frames = 0
        fps_started = time.monotonic()

        mode = (
            "SIMULATION" if arm is None and hand is None else "HARDWARE ARMED / PAUSED"
        )
        print(mode)
        print("Controls: C calibrate/recenter | SPACE live/pause | Q/Esc quit")

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    notice = "Camera read failed"
                    break
                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                now = time.monotonic()

                latest_pose = None
                if result.multi_hand_landmarks:
                    image_hand = result.multi_hand_landmarks[0]
                    world = None
                    if result.multi_hand_world_landmarks:
                        world = result.multi_hand_world_landmarks[0].landmark
                    try:
                        latest_pose = extract_hand_pose(
                            image_hand.landmark,
                            world,
                            finger_calibration=finger_calibration,
                        )
                        last_seen = now
                    except ValueError as exc:
                        notice = f"Unstable hand geometry: {exc}"
                    drawing.draw_landmarks(
                        frame,
                        image_hand,
                        mp.solutions.hands.HAND_CONNECTIONS,
                    )

                if active and now - last_seen > args.loss_timeout:
                    active = False
                    motion.pause("tracking lost; manual resume required")
                    if hand is not None:
                        safe = finger_filter.reset_open()
                        hand.command_fingers(safe)
                        last_sent_angles = safe
                        last_hand_send = now
                    notice = "TRACK LOST: commands paused; show hand and press SPACE"

                state = motion.snapshot()
                if state.status == "error" and active:
                    active = False
                    notice = state.message

                if latest_pose is not None and mapper.calibrated:
                    latest_raw_target = mapper.map(latest_pose)
                    if active:
                        filtered = smoother.update(latest_raw_target)
                        motion.submit(filtered)
                        if hand is not None and now - last_hand_update >= hand_interval:
                            # Keep visual mapping and smoothing independent from
                            # the arm target filter.
                            angles = finger_filter.update(latest_raw_target.closures)
                            last_hand_update = now
                            changed = (
                                last_sent_angles is None
                                or max(
                                    abs(current - previous)
                                    for current, previous in zip(
                                        angles, last_sent_angles
                                    )
                                )
                                >= args.finger_deadband
                            )
                            if changed or now - last_hand_send >= args.finger_keepalive:
                                hand.command_fingers(angles)
                                last_sent_angles = angles
                                last_hand_send = now

                frames += 1
                elapsed = now - fps_started
                if elapsed >= 0.5:
                    fps = frames / elapsed
                    frames = 0
                    fps_started = now

                state = motion.snapshot()
                status_color = (0, 220, 0) if active else (0, 200, 255)
                if state.status in {"rejected", "error"}:
                    status_color = (0, 0, 255)
                lines: list[tuple[str, tuple[int, int, int]]] = [
                    (
                        f"Qmini Avatar | {'LIVE' if active else 'PAUSED'} | "
                        f"{'ARM' if arm is not None else 'arm-sim'} + "
                        f"{'HAND' if hand is not None else 'hand-sim'} | FPS {fps:.1f}",
                        status_color,
                    ),
                    (f"motion: {state.status} | {state.message}", status_color),
                    (notice, (255, 255, 255)),
                    (
                        "C calibrate/recenter | SPACE live/pause | Q quit",
                        (255, 255, 255),
                    ),
                ]
                if latest_pose is not None:
                    closures = " ".join(
                        f"{value:.2f}" for value in latest_pose.closures
                    )
                    lines.append((f"finger closure: {closures}", (255, 220, 80)))
                if latest_raw_target is not None:
                    p = latest_raw_target.position * 1000.0
                    lines.append(
                        (
                            f"target mm: X {p[0]:+.1f}  Y {p[1]:+.1f}  Z {p[2]:+.1f}  "
                            f"wrist {math.degrees(latest_raw_target.servo6):+.1f} deg",
                            (255, 220, 80),
                        )
                    )
                _put_lines(cv2, frame, lines)
                cv2.imshow("Qmini Avatar", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    if latest_pose is None:
                        notice = "Calibration needs a clearly visible hand"
                        continue
                    active = False
                    motion.pause("recalibrated; press SPACE to start")
                    q = motion.snapshot().q_command
                    arm_pose = robot.fk(q)
                    mapper.calibrate(
                        latest_pose,
                        arm_pose[:3, 3],
                        float(q[robot.servo6_index]),
                    )
                    neutral = mapper.map(latest_pose)
                    smoother.reset(neutral)
                    latest_raw_target = neutral
                    notice = "Neutral pose captured; move gently, then press SPACE"
                elif key == 32:
                    if active:
                        active = False
                        motion.pause("operator paused")
                        notice = "PAUSED: servos keep their current goal"
                    elif not mapper.calibrated:
                        notice = "Press C to calibrate before starting"
                    elif latest_pose is None or now - last_seen > args.loss_timeout:
                        notice = "A tracked hand is required to start"
                    elif state.status == "error":
                        notice = "Arm worker has failed; restart the program"
                    else:
                        active = True
                        motion.enable()
                        notice = "LIVE: SPACE pauses immediately"
        finally:
            motion.pause("program exiting")
            capture.release()
            cv2.destroyAllWindows()
            hands.close()

    print("Exited. Serial ports closed; arm torque/last goal may still be active.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        if args.list_ports:
            print_serial_ports()
            return 0
        if args.self_test:
            return run_self_test(args)
        return run_camera(args)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Qmini Avatar stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
