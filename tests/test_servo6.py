"""Independent servo6 control and position-IK contract tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arm_ik import RobotModel
from arm_ik.servo import Servo6Controller, ServoMap

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "description" / "arm.urdf"
CALIBRATION = ROOT / "arm_ik" / "config" / "servo_calibration.yaml"


@pytest.fixture()
def robot() -> RobotModel:
    return RobotModel.from_urdf(URDF)


@pytest.fixture()
def servo_map(robot: RobotModel) -> ServoMap:
    return ServoMap.from_yaml(CALIBRATION, robot.joint_names)


def test_servo6_is_explicitly_separate(robot: RobotModel) -> None:
    assert robot.servo6_index == 5
    assert robot.arm_dof == 5
    assert robot.arm_joint_names == robot.joint_names[:5]
    assert robot.servo6_limits == (robot.lower[5], robot.upper[5])


def test_composing_servo6_keeps_the_position_joints_in_order(
    robot: RobotModel,
) -> None:
    q_arm = np.arange(5, dtype=float) * 0.02
    q = robot.compose_arm_q(q_arm, servo6=0.1)
    assert q.shape == (6,)
    assert np.allclose(q[:5], q_arm)
    assert np.isclose(q[5], 0.1)
    other = robot.compose_arm_q(q_arm, servo6=-0.2)
    assert np.isclose(
        np.linalg.norm(robot.fk(q)[:3, 3] - robot.fk(other)[:3, 3]),
        0.0,
        atol=1e-12,
    )


def test_position_ik_holds_servo6_at_the_requested_angle(robot: RobotModel) -> None:
    q_true = np.array([-0.2, 0.2, 0.9, -0.2, 0.1, 0.0])
    target = robot.fk(q_true)[:3, 3]
    requested = np.radians(20.0)
    result = robot.ik_position(target, servo6=requested)

    assert result.status.is_usable
    assert np.isclose(result.q[robot.servo6_index], requested)
    assert result.position_error < 1e-5


def test_position_ik_accepts_a_five_joint_seed(robot: RobotModel) -> None:
    q_true = np.array([-0.15, 0.2, 0.8, -0.15, 0.1, 0.0])
    target = robot.fk(q_true)[:3, 3]
    seed = robot.mid_range[:5]
    result = robot.ik_position(target, servo6=-0.2, seed=seed)

    assert result.status.is_usable
    assert np.isclose(result.q[5], -0.2)
    assert result.position_error < 1e-5


def test_servo6_controller_maps_and_composes(
    robot: RobotModel,
    servo_map: ServoMap,
) -> None:
    controller = Servo6Controller(robot, servo_map, angle=0.0)
    assert controller.to_tick() == 88

    controller.set_degrees(10.0)
    assert np.isclose(controller.degrees, 10.0)
    assert controller.to_tick() == servo_map.calibrations[5].to_tick(np.radians(10.0))

    q = controller.compose(np.zeros(robot.arm_dof))
    assert np.allclose(q[:5], 0.0)
    assert np.isclose(q[robot.servo6_index], np.radians(10.0))


def test_servo6_controller_rejects_out_of_range_commands(
    robot: RobotModel,
    servo_map: ServoMap,
) -> None:
    controller = Servo6Controller(robot, servo_map)
    with pytest.raises(ValueError, match="servo6"):
        controller.set_degrees(90.0)
    controller.set_degrees(90.0, clamp=True)
    assert np.isclose(controller.degrees, np.degrees(robot.servo6_limits[1]))
    with pytest.raises(ValueError, match="servo6 tick"):
        controller.set_tick(999)
