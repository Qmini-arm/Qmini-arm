from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from qmini_avatar.core import (
    AvatarMapper,
    AvatarMappingConfig,
    AvatarMotionWorker,
    AvatarPlanner,
    AvatarTarget,
    FingerCommandFilter,
    FingerVisionCalibration,
    HumanHandPose,
    PlanningError,
    ReachableWorkspaceProjector,
    SerialPortInfo,
    TargetSmoother,
    choose_avatar_serial_ports,
    extract_hand_pose,
    wrap_angle,
)

from arm_ik import RobotModel
from arm_ik.servo import ServoMap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARM_ROOT = PROJECT_ROOT.parents[1]
URDF = ARM_ROOT / "description" / "arm.urdf"
CALIBRATION = ARM_ROOT / "arm_ik" / "config" / "servo_calibration.yaml"
FINGER_CALIBRATION = PROJECT_ROOT / "config" / "finger_calibration.json"


@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0


def straight_hand() -> list[Point]:
    points = [Point(0.5, 0.5) for _ in range(21)]
    points[0] = Point(0.5, 0.8)
    for index, xy in zip(
        (1, 2, 3, 4),
        ((0.38, 0.72), (0.32, 0.66), (0.26, 0.60), (0.20, 0.54)),
    ):
        points[index] = Point(*xy)
    for base, x in ((5, 0.4), (9, 0.5), (13, 0.6), (17, 0.7)):
        for offset, y in enumerate((0.62, 0.50, 0.38, 0.26)):
            points[base + offset] = Point(x, y)
    return points


@pytest.fixture()
def stack() -> tuple[RobotModel, ServoMap]:
    robot = RobotModel.from_urdf(URDF)
    servo = ServoMap.from_yaml(CALIBRATION, robot.joint_names)
    robot.tighten_limits(*servo.effective_limits(robot))
    return robot, servo


def test_extracts_open_hand_features() -> None:
    pose = extract_hand_pose(straight_hand())
    assert np.allclose(pose.closures, 0.0, atol=1e-8)
    assert pose.palm_scale > 0.1
    assert abs(pose.roll) < 1e-12


def test_natural_fist_reaches_full_finger_closure() -> None:
    points = straight_hand()
    # Give the index finger two 90-degree bends: 180 degrees total is a normal
    # tight curl and should map close to full closure, not the old value 0.5.
    points[5] = Point(0.4, 0.62)
    points[6] = Point(0.4, 0.50)
    points[7] = Point(0.52, 0.50)
    points[8] = Point(0.52, 0.38)
    calibration = FingerVisionCalibration.load(FINGER_CALIBRATION)
    pose = extract_hand_pose(points, finger_calibration=calibration)
    assert pose.closures[1] == 1.0


def test_finger_calibration_maps_captured_endpoints() -> None:
    calibration = FingerVisionCalibration.load(FINGER_CALIBRATION)
    assert np.allclose(calibration.closures_for_scores(calibration.open_scores), 0.0)
    assert np.allclose(calibration.closures_for_scores(calibration.closed_scores), 1.0)
    assert calibration.angles_for_closures((0.0,) * 5) == (180,) * 5
    assert calibration.angles_for_closures((1.0,) * 5) == (0,) * 5


def test_finger_filter_uses_twelve_degree_step() -> None:
    calibration = FingerVisionCalibration.load(FINGER_CALIBRATION)
    finger_filter = FingerCommandFilter(calibration, alpha=0.65, max_step_deg=12.0)
    assert finger_filter.update((1.0,) * 5) == (168,) * 5


def test_relative_mapping_directions_and_hard_bounds() -> None:
    base = HumanHandPose(0.5, 0.5, 0.2, 0.0, (0.0,) * 5)
    mapper = AvatarMapper(
        (-0.5, 0.5),
        AvatarMappingConfig(
            depth_gain_m=0.2,
            lateral_gain_m=0.25,
            vertical_gain_m=0.25,
            max_depth_m=0.05,
            max_lateral_m=0.08,
            max_vertical_m=0.08,
        ),
    )
    mapper.calibrate(base, [0.2, 0.0, 0.18], 0.0)
    moved = HumanHandPose(0.6, 0.4, 0.4, 0.2, (1.0,) * 5)
    target = mapper.map(moved)
    assert np.allclose(target.position, [0.25, -0.025, 0.205])
    assert np.isclose(target.servo6, 0.2)
    assert target.closures == (1.0,) * 5


def test_smoother_limits_each_update() -> None:
    start = AvatarTarget(np.zeros(3), 0.0, (0.0,) * 5)
    goal = AvatarTarget(np.ones(3), math.radians(30), (1.0,) * 5)
    smoother = TargetSmoother(
        alpha=1.0,
        max_position_step_m=0.01,
        max_servo6_step_deg=3.0,
        max_closure_step=0.1,
    )
    smoother.reset(start)
    step = smoother.update(goal)
    assert np.allclose(step.position, 0.01)
    assert np.isclose(step.servo6, math.radians(3.0))
    assert np.allclose(step.closures, 0.1)


def test_neutral_pose_plans_without_motion(stack: tuple[RobotModel, ServoMap]) -> None:
    robot, servo = stack
    planner = AvatarPlanner(robot, servo)
    q0 = robot.mid_range
    target = AvatarTarget(robot.fk(q0)[:3, 3], float(q0[5]), (0.0,) * 5)
    planned = planner.plan(q0, target)
    assert planned.position_error < 1e-5
    assert planned.max_joint_step_deg < 1e-3


def test_large_wrist_change_is_slew_limited_instead_of_rejected(
    stack: tuple[RobotModel, ServoMap],
) -> None:
    robot, servo = stack
    q0 = robot.mid_range
    planner = AvatarPlanner(robot, servo, max_joint_step_deg=0.5)
    target = AvatarTarget(robot.fk(q0)[:3, 3], float(robot.upper[5]), (0.0,) * 5)
    planned = planner.plan(q0, target)
    assert planned.slew_limited
    assert planned.max_joint_step_deg <= 0.5 + 1e-9


def test_unreachable_request_is_projected_to_known_workspace(
    stack: tuple[RobotModel, ServoMap],
) -> None:
    robot, servo = stack
    q0 = robot.mid_range
    position = robot.fk(q0)[:3, 3]
    workspace = ReachableWorkspaceProjector(
        robot,
        np.repeat(q0[None, :], 10, axis=0),
        np.repeat(position[None, :], 10, axis=0),
        coverage_tolerance_m=0.001,
    )
    planner = AvatarPlanner(robot, servo, workspace=workspace)
    requested = AvatarTarget(np.array([1.0, 1.0, 1.0]), float(q0[5]), (0.0,) * 5)
    planned = planner.plan(q0, requested)
    assert planned.projected
    assert planned.projection_distance > 0.5
    assert np.allclose(planned.target.position, position)
    assert planned.position_error < 1e-5


def test_collision_rejects_before_command(stack: tuple[RobotModel, ServoMap]) -> None:
    robot, servo = stack

    class AlwaysColliding:
        def check(self, _q: np.ndarray) -> list[str]:
            return ["synthetic collision"]

    planner = AvatarPlanner(robot, servo, collision_checker=AlwaysColliding())  # type: ignore[arg-type]
    q0 = robot.mid_range
    target = AvatarTarget(robot.fk(q0)[:3, 3], float(q0[5]), (0.0,) * 5)
    with pytest.raises(PlanningError, match="self collision"):
        planner.plan(q0, target)


def test_motion_worker_runs_without_hardware(
    stack: tuple[RobotModel, ServoMap]
) -> None:
    robot, servo = stack
    planner = AvatarPlanner(robot, servo)
    q0 = robot.mid_range
    target = AvatarTarget(robot.fk(q0)[:3, 3], float(q0[5]), (0.0,) * 5)
    worker = AvatarMotionWorker(planner, q0, rate_hz=30.0)
    try:
        worker.enable()
        worker.submit(target)
        deadline = time.monotonic() + 1.0
        while worker.snapshot().sent_count == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        state = worker.snapshot()
        assert state.status == "sim"
        assert state.sent_count == 1
    finally:
        worker.close()


def test_wrap_angle_handles_branch_cut() -> None:
    assert np.isclose(wrap_angle(3 * math.pi), math.pi)
    assert np.isclose(wrap_angle(-3 * math.pi), -math.pi)


def test_joint_serial_detection_assigns_distinct_roles() -> None:
    ports = [
        SerialPortInfo("/dev/cu.usbserial-1410", "USB Serial", "FTDI"),
        SerialPortInfo("/dev/cu.usbmodem1101", "Arduino Uno", "USB VID:PID"),
    ]
    arm, hand = choose_avatar_serial_ports(ports, want_arm=True, want_hand=True)
    assert arm == "/dev/cu.usbserial-1410"
    assert hand == "/dev/cu.usbmodem1101"


def test_serial_detection_refuses_ambiguous_arm_ports() -> None:
    ports = [
        SerialPortInfo("/dev/cu.usbserial-A", "USB Serial", "FTDI"),
        SerialPortInfo("/dev/cu.usbserial-B", "USB Serial", "FTDI"),
        SerialPortInfo("/dev/cu.usbmodem1101", "Arduino Uno", ""),
    ]
    with pytest.raises(RuntimeError, match="ambiguous arm"):
        choose_avatar_serial_ports(ports, want_arm=True, want_hand=True)


def test_serial_detection_never_reuses_an_explicit_port() -> None:
    port = "/dev/cu.usbmodem1101"
    with pytest.raises(RuntimeError, match="same serial port"):
        choose_avatar_serial_ports(
            [SerialPortInfo(port, "Arduino Uno", "")],
            want_arm=True,
            want_hand=True,
            arm_override=port,
            hand_override=port,
        )
