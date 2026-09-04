#!/usr/bin/env python3
"""Camera RPS recognition with a winning response from the Qmini uHand."""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from qmini_rps.core import (
    Gesture,
    StableGestureRecognizer,
    classify_closures,
    winning_command,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINGER_CALIBRATION = (
    PROJECT_ROOT.parent / "qmini_avatar" / "config" / "finger_calibration.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recognize rock/paper/scissors and make the Qmini uHand win",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument(
        "--camera-format",
        choices=("mjpg", "auto"),
        default="mjpg",
        help="capture pixel format; MJPG avoids corrupted YUYV frames over WSL USB/IP",
    )
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument(
        "--live",
        action="store_true",
        help="connect an auto-detected uHand USB port; still starts paused",
    )
    parser.add_argument(
        "--hand-port",
        default=None,
        help="explicit uHand USB port; providing it enables hardware mode",
    )
    parser.add_argument(
        "--finger-calibration",
        type=Path,
        default=DEFAULT_FINGER_CALIBRATION,
        help="reuse a qmini_avatar-compatible finger calibration JSON",
    )
    parser.add_argument("--stable-frames", type=int, default=5)
    parser.add_argument("--open-threshold", type=float, default=0.35)
    parser.add_argument("--closed-threshold", type=float, default=0.65)
    parser.add_argument("--loss-timeout", type=float, default=0.6)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="test classification and response mapping without camera or hardware",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0:
        raise ValueError("camera width and height must be positive")
    if args.stable_frames < 1:
        raise ValueError("stable-frames must be at least 1")
    if not 0.0 <= args.open_threshold < args.closed_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= open < closed <= 1")
    if args.loss_timeout <= 0:
        raise ValueError("loss-timeout must be positive")


def run_self_test() -> int:
    examples = {
        Gesture.ROCK: (0.9, 0.9, 0.9, 0.9, 0.9),
        Gesture.PAPER: (0.1, 0.1, 0.1, 0.1, 0.1),
        Gesture.SCISSORS: (0.5, 0.1, 0.1, 0.9, 0.9),
    }
    expected = {
        Gesture.ROCK: Gesture.PAPER,
        Gesture.PAPER: Gesture.SCISSORS,
        Gesture.SCISSORS: Gesture.ROCK,
    }
    for human, closures in examples.items():
        prediction = classify_closures(closures)
        assert prediction is not None and prediction.gesture is human
        command = winning_command(human)
        assert command.robot is expected[human]
        print(f"human={human.value:8s} -> robot={command.robot.value:8s} ({command.uhand_gesture})")
    print("Qmini RPS self-test passed; no camera or serial port was opened")
    return 0


def _put_lines(
    cv2: Any, frame: Any, lines: list[tuple[str, tuple[int, int, int]]]
) -> None:
    y = 28
    for label, color in lines:
        cv2.putText(
            frame,
            label,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 27


def _send_preset(hand: Any | None, preset: str) -> None:
    if hand is not None:
        # The uHand firmware applies its own 20 ms low-pass response. A zero-
        # duration API command avoids blocking the camera loop with host-side
        # interpolation while still going through the validated public API.
        hand.gesture(preset, duration=0.0)


def configure_capture(
    cv2: Any,
    capture: Any,
    width: int,
    height: int,
    camera_format: str,
) -> None:
    """Negotiate a USB/IP-friendly format before setting capture dimensions."""

    if camera_format == "mjpg":
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def run_camera(args: argparse.Namespace) -> int:
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
        from qmini_avatar.core import FingerVisionCalibration, extract_hand_pose
    except ImportError as exc:
        print(f"Missing camera dependency: {exc}", file=sys.stderr)
        print("Run `uv sync` in demos/qmini_rps first.", file=sys.stderr)
        return 2

    calibration = FingerVisionCalibration.load(args.finger_calibration)
    recognizer = StableGestureRecognizer(args.stable_frames)
    capture = cv2.VideoCapture(args.camera)
    configure_capture(cv2, capture, args.width, args.height, args.camera_format)
    if not capture.isOpened():
        capture.release()
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 2

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    drawing = mp.solutions.drawing_utils
    use_hardware = bool(args.live or args.hand_port is not None)

    with ExitStack() as stack:
        hand = None
        if use_hardware:
            from uhand import connect

            port = args.hand_port or "auto"
            print(f"Connecting uHand on {port}...")
            hand = stack.enter_context(connect(port))
            print(f"Connected uHand: {hand.port}; commands are PAUSED")

        active = False
        last_seen = 0.0
        tracking_lost = True
        latest_update = recognizer.update(None)
        latest_closures: tuple[float, ...] | None = None
        robot_gesture: Gesture | None = None
        notice = "Show rock/paper/scissors, then press SPACE to start"
        mode = "UHAND" if hand is not None else "SIMULATION"
        print(f"{mode} / PAUSED")
        print("Controls: SPACE live/pause | R reset/open | Q/Esc quit")

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

                prediction = None
                if result.multi_hand_landmarks:
                    image_hand = result.multi_hand_landmarks[0]
                    world = None
                    if result.multi_hand_world_landmarks:
                        world = result.multi_hand_world_landmarks[0].landmark
                    try:
                        pose = extract_hand_pose(
                            image_hand.landmark,
                            world,
                            finger_calibration=calibration,
                        )
                        latest_closures = pose.closures
                        prediction = classify_closures(
                            pose.closures,
                            open_threshold=args.open_threshold,
                            closed_threshold=args.closed_threshold,
                        )
                        last_seen = now
                        tracking_lost = False
                    except ValueError as exc:
                        latest_closures = None
                        notice = f"Unstable hand geometry: {exc}"
                    drawing.draw_landmarks(
                        frame,
                        image_hand,
                        mp.solutions.hands.HAND_CONNECTIONS,
                    )

                latest_update = recognizer.update(prediction)
                if latest_update.changed and latest_update.stable is not None:
                    command = winning_command(latest_update.stable)
                    robot_gesture = command.robot
                    notice = (
                        f"Human {command.human.value}; robot answers {command.robot.value}"
                    )
                    if active:
                        try:
                            _send_preset(hand, command.uhand_gesture)
                        except Exception as exc:
                            active = False
                            notice = f"uHand command failed: {exc}"

                if not tracking_lost and now - last_seen > args.loss_timeout:
                    tracking_lost = True
                    latest_closures = None
                    recognizer.reset(clear_stable=True)
                    latest_update = recognizer.update(None)
                    if active:
                        active = False
                        try:
                            _send_preset(hand, "open")
                            robot_gesture = Gesture.PAPER
                            notice = "TRACK LOST: hand opened; press SPACE to resume"
                        except Exception as exc:
                            notice = f"TRACK LOST; uHand open failed: {exc}"

                status_color = (0, 220, 0) if active else (0, 200, 255)
                observed = (
                    "uncertain"
                    if latest_update.observed is None
                    else latest_update.observed.value
                )
                stable = (
                    "none" if latest_update.stable is None else latest_update.stable.value
                )
                robot = "none" if robot_gesture is None else robot_gesture.value
                lines = [
                    (
                        f"Qmini RPS | {'LIVE' if active else 'PAUSED'} | {mode}",
                        status_color,
                    ),
                    (
                        f"seen: {observed} ({latest_update.streak}/{args.stable_frames}) "
                        f"| stable: {stable}",
                        (255, 220, 80),
                    ),
                    (f"robot: {robot}", (80, 255, 120)),
                    (notice, (255, 255, 255)),
                    ("SPACE live/pause | R reset/open | Q quit", (255, 255, 255)),
                ]
                if latest_closures is not None:
                    values = " ".join(f"{value:.2f}" for value in latest_closures)
                    lines.append((f"closure T I M R P: {values}", (255, 220, 80)))
                _put_lines(cv2, frame, lines)
                cv2.imshow("Qmini Rock Paper Scissors", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == 32:
                    if active:
                        active = False
                        notice = "PAUSED: uHand keeps its current target"
                    elif tracking_lost or latest_update.stable is None:
                        notice = "Show a stable RPS hand before starting"
                    else:
                        active = True
                        command = winning_command(latest_update.stable)
                        robot_gesture = command.robot
                        try:
                            _send_preset(hand, command.uhand_gesture)
                            notice = "LIVE: show another gesture; SPACE pauses"
                        except Exception as exc:
                            active = False
                            notice = f"uHand command failed: {exc}"
                elif key == ord("r"):
                    active = False
                    recognizer.reset(clear_stable=True)
                    latest_update = recognizer.update(None)
                    robot_gesture = Gesture.PAPER
                    try:
                        _send_preset(hand, "open")
                        notice = "Reset: uHand opened; show a gesture and press SPACE"
                    except Exception as exc:
                        notice = f"Reset; uHand open failed: {exc}"
        finally:
            capture.release()
            cv2.destroyAllWindows()
            hands.close()

    print("Exited. Serial port closed; uHand may keep its last target.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        if args.self_test:
            return run_self_test()
        return run_camera(args)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Qmini RPS stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
