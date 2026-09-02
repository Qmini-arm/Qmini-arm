"""Tests for the optional live-arm adapter used by the Viser controls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arm_ik import RobotModel
from arm_ik.servo.mapping import ServoMap
from arm_ik.viz.viser_app import _build_real_arm_driver, _RealArmDriver

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "description" / "arm.urdf"
CALIBRATION = ROOT / "arm_ik" / "config" / "servo_calibration.yaml"


class FakeCommandBackend:
    port_name = "fake"

    def __init__(self, servo_map: ServoMap) -> None:
        self.servo_map = servo_map
        self.calls: list[tuple[object, ...]] = []

    def read_positions(self) -> dict[int, int]:
        self.calls.append(("read",))
        return {
            cal.servo_id: cal.center_tick for cal in self.servo_map.calibrations
        }

    def takeover_current(self, *, speed: int = 160) -> dict[int, int]:
        self.calls.append(("takeover", speed))
        return self.read_positions()

    def send_goals(
        self,
        positions: dict[int, int],
        speed: int,
        *,
        verify: bool = True,
    ) -> bytes:
        self.calls.append(("send", dict(positions), speed, verify))
        return b""


@pytest.fixture()
def robot() -> RobotModel:
    return RobotModel.from_urdf(URDF)


@pytest.fixture()
def servo_map(robot: RobotModel) -> ServoMap:
    return ServoMap.from_yaml(CALIBRATION, robot.joint_names)


def test_driver_is_read_only_until_takeover_and_enable(
    robot: RobotModel, servo_map: ServoMap
) -> None:
    backend = FakeCommandBackend(servo_map)
    driver = _RealArmDriver(backend, servo_map, speed=111)
    q = driver.read_current()

    assert np.allclose(q, np.zeros(robot.dof))
    assert backend.calls == [("read",)]
    assert not driver.command(q)

    driver.takeover()
    assert backend.calls[-2:] == [("takeover", 111), ("read",)]
    assert not driver.command(q)

    driver.enable()
    moved = q.copy()
    moved[0] += 0.01
    assert driver.command(moved)
    assert backend.calls[-1][0] == "send"
    assert backend.calls[-1][-1] is False
    assert not driver.command(moved), "unchanged targets should not spam the bus"

    driver.disable()
    moved[0] += 0.01
    assert not driver.command(moved)


def test_live_driver_tightens_ik_limits_to_servo_window(
    robot: RobotModel, servo_map: ServoMap
) -> None:
    backend = FakeCommandBackend(servo_map)
    expected_lower, expected_upper = servo_map.effective_limits(robot)
    original_lower = robot.lower.copy()
    driver = _build_real_arm_driver(robot, backend, speed=160)

    assert driver is not None
    assert np.allclose(robot.lower, expected_lower)
    assert np.allclose(robot.upper, expected_upper)
    assert np.any(robot.lower > original_lower)


def test_live_driver_rejects_read_only_backend(robot: RobotModel) -> None:
    class ReadOnlyBackend:
        def read_positions(self) -> dict[int, int]:
            return {}

    with pytest.raises(TypeError, match="可写"):
        _build_real_arm_driver(robot, ReadOnlyBackend(), speed=160)


def test_live_driver_validates_speed(robot: RobotModel) -> None:
    servo_map = ServoMap.from_yaml(CALIBRATION, robot.joint_names)
    backend = FakeCommandBackend(servo_map)
    with pytest.raises(ValueError, match="1..1023"):
        _RealArmDriver(backend, servo_map, speed=0)


def test_cli_viewer_auto_connects_without_device(monkeypatch: pytest.MonkeyPatch) -> None:
    import arm_ik.cli as cli
    import arm_ik.viz as viz
    import cds_arm

    backend = object()
    calls: list[tuple[str, object]] = []

    class Connection:
        def __enter__(self) -> object:
            calls.append(("enter", backend))
            return backend

        def __exit__(self, *_: object) -> None:
            calls.append(("exit", backend))

    def fake_connect(*args: object) -> Connection:
        calls.append(("connect", args))
        return Connection()

    def fake_launch(*args: object, **kwargs: object) -> None:
        calls.append(("launch", kwargs))

    monkeypatch.setattr(cds_arm, "connect", fake_connect)
    monkeypatch.setattr(viz, "launch_viewer", fake_launch)

    assert cli.main(["--urdf", str(URDF), "viz", "--mode", "viewer"]) == 0
    assert calls[0] == ("connect", ())
    assert calls[1] == ("enter", backend)
    assert calls[2][0] == "launch"
    assert calls[2][1]["servo_backend"] is backend
    assert calls[3] == ("exit", backend)


def test_cli_viz_can_select_device_or_stay_in_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import arm_ik.cli as cli
    import arm_ik.viz as viz
    import cds_arm

    calls: list[tuple[str, object]] = []

    class Connection:
        def __enter__(self) -> object:
            return "backend"

        def __exit__(self, *_: object) -> None:
            calls.append(("exit", None))

    def fake_connect(*args: object) -> Connection:
        calls.append(("connect", args))
        return Connection()

    def fake_launch(*args: object, **kwargs: object) -> None:
        calls.append(("launch", kwargs))

    monkeypatch.setattr(cds_arm, "connect", fake_connect)
    monkeypatch.setattr(viz, "launch_ik_app", fake_launch)

    assert (
        cli.main(
            [
                "--urdf",
                str(URDF),
                "viz",
                "--mode",
                "ik",
                "--device",
                "/dev/test-arm",
            ]
        )
        == 0
    )
    assert calls[0] == ("connect", ("/dev/test-arm",))
    assert calls[1][0] == "launch"
    assert calls[1][1]["servo_backend"] == "backend"

    calls.clear()
    assert cli.main(["--urdf", str(URDF), "viz", "--mode", "ik", "--sim"]) == 0
    assert calls == [("launch", {"host": "0.0.0.0", "port": 8080})]
