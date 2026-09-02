"""Viser browser visualisation and FK simulation for the arm.

Three entry points share one ``ViserUrdf`` render path:

- :func:`launch_viewer` -- joint-angle sliders over the URDF's own limits, with
  an optional explicitly enabled real-arm command path.
- :func:`launch_ik_app` -- an end-effector target driving IK, adjustable either
  by dragging a gizmo or by six sliders (x/y/z and roll/pitch/yaw), with the
  same optional real-arm command path.
- :func:`replay` -- read the real servos and animate the digital twin.

viser renders in a browser; the server is a local Python process, so the same
viewer can run on a headless machine and be watched from a laptop, which makes
it useful for debugging the real arm.

Scene nodes are created once and then mutated in place. Calling ``add_frame``
again with the same name replaces the node and invalidates the previous handle,
and because viser dispatches GUI callbacks on a thread pool, two overlapping
re-adds leave one of them writing to a handle that has already been removed.
Rebuilding a node per drag event is also what makes the viewer stutter.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from ..config import IKStatus
from ..model.transforms import FloatArray, matrix_to_rpy, rpy_to_matrix
from ..servo.mapping import ServoBackend, ServoMap

if TYPE_CHECKING:
    from ..model.robot_model import RobotModel

logger = logging.getLogger(__name__)

__all__ = ["launch_viewer", "launch_ik_app", "replay"]

# This arm reaches about 0.35 m, so viser's 0.5 m default axes would dwarf it.
_AXES_LENGTH = 0.06
_AXES_RADIUS = 0.003

# Sample count for deriving slider ranges. Enough to cover the reachable shell
# without adding a noticeable pause at startup.
_BOUNDS_SAMPLES = 1500
_BOUNDS_MARGIN = 0.05

# The reachable set is a thin shell, so it needs a fair number of points to read
# as a surface rather than a scatter. 8000 samples take well under a second.
_CLOUD_SAMPLES = 8000
_CLOUD_POINT_SIZE = 0.002


class _CommandBackend(Protocol):
    """The writable subset of a real-arm backend used by the viewer.

    ``ServoBackend`` intentionally only promises feedback reads because replay
    does not need a command path.  The live controls require the two additional
    CDSArm operations below, so keep that stronger contract local to this
    module instead of making read-only backends implement fake writes.
    """

    def read_positions(self) -> dict[int, int]: ...

    def takeover_current(self, *, speed: int = 160) -> dict[int, int]: ...

    def send_goals(
        self,
        positions: dict[int, int],
        speed: int,
        *,
        verify: bool = True,
    ) -> bytes: ...


class _RealArmDriver:
    """Translate joint configurations into guarded, latest-state bus writes."""

    def __init__(
        self,
        backend: _CommandBackend,
        servo_map: ServoMap,
        *,
        speed: int = 160,
    ) -> None:
        self.backend = backend
        self.servo_map = servo_map
        self._lock = threading.RLock()
        self._enabled = False
        self._armed = False
        self._last_ticks: dict[int, int] | None = None
        self._speed = 160
        self.speed = speed

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def speed(self) -> int:
        return self._speed

    @speed.setter
    def speed(self, value: int | float) -> None:
        speed = int(round(float(value)))
        if not 1 <= speed <= 1023:
            raise ValueError("实机速度必须在1..1023")
        self._speed = speed

    @property
    def port_name(self) -> str:
        return str(getattr(self.backend, "port_name", "串口"))

    def read_current(self) -> FloatArray:
        """Read feedback without writing or changing torque state."""
        with self._lock:
            if self._enabled:
                raise RuntimeError("实机驱动启用时不能单独读取当前位置")
            ticks = self.backend.read_positions()
            self._last_ticks = dict(ticks)
            return self.servo_map.to_joints(ticks)

    def takeover(self) -> FloatArray:
        """Hold the present pose, then return it in URDF joint coordinates.

        ``CDSArm.takeover_current`` samples a stable feedback pose and writes
        that same pose before enabling torque.  The driver remains disabled until
        the caller has synchronised the browser controls to the returned pose.
        """
        with self._lock:
            self._enabled = False
            ticks = self.backend.takeover_current(speed=self._speed)
            self._last_ticks = dict(ticks)
            self._armed = True
            return self.servo_map.to_joints(ticks)

    def enable(self) -> None:
        with self._lock:
            if not self._armed:
                raise RuntimeError("启用实机驱动前必须先接管当前位置")
            self._enabled = True

    def disable(self) -> None:
        """Stop sending new goals; existing servo torque is intentionally kept."""
        with self._lock:
            self._enabled = False
            self._armed = False

    def command(self, q: FloatArray) -> bool:
        """Send a changed joint configuration, returning whether it was sent."""
        with self._lock:
            if not self._enabled:
                return False
            angles = np.asarray(q, dtype=float).reshape(-1)
            if not np.all(np.isfinite(angles)):
                raise ValueError("不能向实机发送非有限关节角")
            ticks = self.servo_map.to_ticks(angles)
            if ticks == self._last_ticks:
                return False
            # Position feedback is already validated by takeover and the CDSArm
            # API validates every goal against its independent safety window.
            self.backend.send_goals(ticks, self._speed, verify=False)
            self._last_ticks = dict(ticks)
            return True


def _calibration_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "servo_calibration.yaml"


def _build_real_arm_driver(
    robot: RobotModel,
    servo_backend: ServoBackend | None,
    *,
    speed: int,
) -> _RealArmDriver | None:
    """Prepare the optional live driver and fold its limits into the model."""
    if servo_backend is None:
        return None
    missing = [
        name
        for name in ("takeover_current", "send_goals")
        if not callable(getattr(servo_backend, name, None))
    ]
    if missing:
        raise TypeError(
            "实机驱动需要可写的CDSArm后端，缺少: " + ", ".join(missing)
        )

    servo_map = ServoMap.from_yaml(_calibration_path(), robot.joint_names)
    for problem in servo_map.validate_against(robot):
        logger.warning("实机标定检查: %s", problem)
    lower, upper = servo_map.effective_limits(robot)
    robot.tighten_limits(lower, upper)
    return _RealArmDriver(cast(_CommandBackend, servo_backend), servo_map, speed=speed)


class _RealArmControls:
    """Shared Viser controls for the optional real-arm command path."""

    def __init__(
        self,
        server: Any,
        driver: _RealArmDriver | None,
        lock: threading.RLock,
        on_sync: Callable[[FloatArray], None],
    ) -> None:
        self.driver = driver
        self._lock = lock
        self._on_sync = on_sync
        self._syncing = False
        self._busy = False

        folder = server.gui.add_folder("实际机械臂")
        with folder:
            self.drive = server.gui.add_checkbox(
                "驱动实际机械臂",
                initial_value=False,
            )
            self.speed = server.gui.add_slider(
                "速度",
                min=1,
                max=1023,
                step=1,
                initial_value=float(driver.speed if driver is not None else 160),
            )
            self.read = server.gui.add_button("读取当前位置")
            self.status = server.gui.add_text(
                "实机状态",
                initial_value=(
                    "已连接，未接管"
                    if driver is not None
                    else "未连接串口（启动时传入 --device/--serial）"
                ),
            )

        self.drive.on_update(self._on_drive)
        self.speed.on_update(self._on_speed)
        self.read.on_click(self._on_read)
        if driver is None:
            self.drive.disabled = True
            self.speed.disabled = True
            self.read.disabled = True

    def _set_drive_value(self, value: bool) -> None:
        self._syncing = True
        try:
            self.drive.value = value
        finally:
            self._syncing = False

    def _on_speed(self, _args: object = None) -> None:
        with self._lock:
            if self.driver is None or self._syncing:
                return
            try:
                self.driver.speed = self.speed.value
            except Exception as exc:
                self.status.value = f"速度设置失败: {exc}"

    def _on_read(self, _args: object = None) -> None:
        with self._lock:
            if self.driver is None or self._busy:
                return
            if self.driver.enabled:
                self.status.value = "请先关闭实机驱动，再读取当前位置"
                return
            self._busy = True
            self.read.disabled = True
            try:
                q = self.driver.read_current()
                self._on_sync(q)
                self.status.value = "已读取当前位置（尚未发送目标）"
            except Exception as exc:
                self.status.value = f"读取失败: {exc}"
                logger.exception("读取实机当前位置失败")
            finally:
                self.read.disabled = False
                self._busy = False

    def _on_drive(self, _args: object = None) -> None:
        with self._lock:
            if self.driver is None or self._syncing or self._busy:
                return
            if not bool(self.drive.value):
                self.driver.disable()
                self.status.value = "已停止发送（舵机扭矩保持）"
                return

            self._busy = True
            self.drive.disabled = True
            self.speed.disabled = True
            self.read.disabled = True
            try:
                # Keep the driver disabled while the callback aligns the browser
                # controls; slider callbacks therefore cannot issue a jump.
                q = self.driver.takeover()
                self._on_sync(q)
                self.driver.enable()
                self.status.value = f"实机驱动已启用 ({self.driver.port_name})"
            except Exception as exc:
                self.driver.disable()
                self._set_drive_value(False)
                self.status.value = f"接管失败: {exc}"
                logger.exception("接管实机失败")
            finally:
                self.drive.disabled = False
                self.speed.disabled = False
                self.read.disabled = False
                self._busy = False

    def report_command_error(self, exc: Exception) -> None:
        """Disable the live path after a bus failure and expose the reason."""
        with self._lock:
            if self.driver is not None:
                self.driver.disable()
            self._set_drive_value(False)
            self.status.value = f"发送失败，已停用: {exc}"
            logger.exception("发送实机关节目标失败")


def _load_viser_model(
    robot: RobotModel,
    *,
    load_collision: bool = False,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> tuple[Any, Any]:
    """Build a server plus a ``ViserUrdf`` bound to it.

    viser is untyped, so the handles are returned as :data:`Any`. Their dynamic
    methods (``get_actuated_joint_limits``, ``update_cfg``, ``scene``, ...) are
    still verified at runtime; this annotation only silences the false statically
    inferred ``object`` type that would otherwise reject every method call.
    """
    import viser
    from viser.extras import ViserUrdf

    server = viser.ViserServer(host=host, port=port)
    # ViserUrdf accepts a Path or a yourdfpy.URDF, never a str: a str falls
    # through its type dispatch and trips a bare assert.
    model = ViserUrdf(
        server,
        Path(robot.urdf.source),
        load_collision_meshes=load_collision,
    )
    return server, model


def _joint_order(model: Any) -> tuple[str, ...]:
    """Joint names in the order ``update_cfg`` expects."""
    return tuple(model.get_actuated_joint_names())


def _add_tip_frame(server: Any, name: str, pose: FloatArray) -> Any:
    """Create a coordinate frame sized for this arm. Call once per node name."""
    from viser import transforms as tf

    return server.scene.add_frame(
        name,
        wxyz=tf.SO3.from_matrix(pose[:3, :3]).wxyz,
        position=pose[:3, 3],
        axes_length=_AXES_LENGTH,
        axes_radius=_AXES_RADIUS,
    )


def _update_frame(handle: Any, pose: FloatArray) -> None:
    """Move an existing frame. Cheaper than re-adding, and thread-safe."""
    from viser import transforms as tf

    handle.wxyz = tf.SO3.from_matrix(pose[:3, :3]).wxyz
    handle.position = pose[:3, 3]


def _position_bounds(robot: RobotModel) -> tuple[FloatArray, FloatArray]:
    """Reachable position bounds, widened a little, for slider ranges.

    Derived by sampling rather than hard-coded so the sliders still span the
    workspace if the URDF limits change. The margin lets the user push slightly
    past the reachable shell, where IK reports being out of reach instead of
    silently pinning at the edge.
    """
    from ..workspace import sample_workspace

    positions = sample_workspace(robot, count=_BOUNDS_SAMPLES, seed=0).positions
    lower = positions.min(axis=0) - _BOUNDS_MARGIN
    upper = positions.max(axis=0) + _BOUNDS_MARGIN
    return lower, upper


def _cloud_colors(positions: FloatArray) -> np.ndarray:
    """Colour the shell by distance from the base, so its depth is legible."""
    radii = np.linalg.norm(positions, axis=1)
    span = float(radii.max() - radii.min())
    t = (radii - radii.min()) / max(span, 1e-9)
    colors = np.empty((len(positions), 3), dtype=np.uint8)
    colors[:, 0] = (40 + 60 * t).astype(np.uint8)
    colors[:, 1] = (90 + 150 * t).astype(np.uint8)
    colors[:, 2] = (140 + 100 * t).astype(np.uint8)
    return colors


def _add_workspace_toggle(server: Any, robot: RobotModel) -> Any:
    """Add a checkbox that shows the reachable workspace as a point cloud.

    The cloud is sampled on first use rather than at startup, so the viewer opens
    immediately and only pays for the sampling if the user asks to see it. Seeing
    the shell is the quickest way to understand why a target is unreachable:
    this arm's reachable set is a curved surface, not a solid volume.
    """
    checkbox = server.gui.add_checkbox("显示可达空间", initial_value=False)
    state: dict[str, Any] = {"handle": None}
    lock = threading.RLock()

    def on_toggle(_args: object = None) -> None:
        with lock:
            if state["handle"] is None:
                if not checkbox.value:
                    return
                from ..workspace import sample_workspace

                checkbox.disabled = True
                try:
                    points = sample_workspace(
                        robot, count=_CLOUD_SAMPLES, seed=0
                    ).positions
                    state["handle"] = server.scene.add_point_cloud(
                        "/reachable_workspace",
                        points=points.astype(np.float32),
                        colors=_cloud_colors(points),
                        point_size=_CLOUD_POINT_SIZE,
                    )
                finally:
                    checkbox.disabled = False
                return
            state["handle"].visible = bool(checkbox.value)

    checkbox.on_update(on_toggle)
    return checkbox


def launch_viewer(
    robot: RobotModel,
    q0: FloatArray | None = None,
    *,
    servo_backend: ServoBackend | None = None,
    speed: int = 160,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Open a browser viewer with a slider per actuated joint.

    Dragging a slider updates the render immediately. The end-effector pose is
    shown as a coordinate frame with its position printed in a label. When
    ``servo_backend`` is supplied, the panel also offers an explicitly gated
    real-arm drive path; it is disabled until the operator enables it in Viser.
    """
    driver = _build_real_arm_driver(robot, servo_backend, speed=speed)
    server, model = _load_viser_model(robot, host=host, port=port)
    names = _joint_order(model)
    if driver is None:
        limits = model.get_actuated_joint_limits()
    else:
        limits = {
            name: (float(robot.lower[i]), float(robot.upper[i]))
            for i, name in enumerate(robot.joint_names)
        }

    start = robot.mid_range if q0 is None else robot.clamp(q0)
    sliders: dict[str, Any] = {}
    lock = threading.RLock()
    syncing = {"active": False}
    real_controls: _RealArmControls | None = None

    tip = robot.fk(start)
    tip_frame = _add_tip_frame(server, "/tip_pose", tip)
    tip_label = server.scene.add_label(
        "/tip_label",
        f"tip: ({tip[0, 3]:.3f}, {tip[1, 3]:.3f}, {tip[2, 3]:.3f}) m",
        position=tip[:3, 3] + np.array([0.0, 0.0, 0.05]),
    )

    def render(q: FloatArray) -> None:
        model.update_cfg(q)
        pose = robot.fk(q)
        _update_frame(tip_frame, pose)
        tip_label.text = (
            f"tip: ({pose[0, 3]:.3f}, {pose[1, 3]:.3f}, {pose[2, 3]:.3f}) m"
        )
        tip_label.position = pose[:3, 3] + np.array([0.0, 0.0, 0.05])

    def on_update(_args: object = None) -> None:
        with lock:
            if syncing["active"]:
                return
            q = np.array([float(sliders[name].value) for name in names])
            render(q)
            if driver is not None:
                try:
                    driver.command(q)
                except Exception as exc:
                    if real_controls is not None:
                        real_controls.report_command_error(exc)
                    else:
                        logger.exception("发送实机关节目标失败")

    for name in names:
        lo, hi = limits[name]
        lo = -np.pi if lo is None else lo
        hi = np.pi if hi is None else hi
        slider = server.gui.add_slider(
            name,
            min=lo,
            max=hi,
            step=0.005,
            initial_value=float(start[robot.joint_names.index(name)]),
        )
        slider.on_update(on_update)
        sliders[name] = slider

    def sync_from_q(q: FloatArray) -> None:
        """Align the browser controls to a feedback pose without commanding it."""
        with lock:
            q = robot.clamp(q)
            syncing["active"] = True
            try:
                for name in names:
                    sliders[name].value = float(q[robot.joint_names.index(name)])
            finally:
                syncing["active"] = False
            render(q)

    real_controls = _RealArmControls(server, driver, lock, sync_from_q)
    _add_workspace_toggle(server, robot)

    on_update()
    logger.info("viser viewer 启动: http://%s:%d", host, port)
    while True:
        time.sleep(1.0)


def launch_ik_app(
    robot: RobotModel,
    *,
    servo_backend: ServoBackend | None = None,
    speed: int = 160,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Open a viewer where an end-effector target drives IK.

    The target can be moved two ways, kept in sync: a draggable gizmo, and six
    sliders (position in metres, orientation as roll/pitch/yaw in degrees).
    Sliders are the easier control for fine adjustment and for setting one axis
    at a time; the gizmo is faster for coarse moves.

    Solver status is shown in the panel, including a separate note when the
    position is reachable but the requested orientation is not -- the common
    case on this arm, whose wrist cone is narrow.

    When ``servo_backend`` is supplied, enabling the real-arm control in the
    panel first takes over the current feedback pose and only then sends later
    IK solutions.
    """
    from viser import transforms as tf

    driver = _build_real_arm_driver(robot, servo_backend, speed=speed)
    server, model = _load_viser_model(robot, host=host, port=port)

    q0 = robot.mid_range
    tip = robot.fk(q0)
    lower, upper = _position_bounds(robot)

    gizmo = server.scene.add_transform_controls(
        "/target_tf",
        position=tip[:3, 3],
        wxyz=tf.SO3.from_matrix(tip[:3, :3]).wxyz,
        depth_test=False,
        scale=0.15,
    )
    reached_frame = _add_tip_frame(server, "/reached_pose", tip)

    position_folder = server.gui.add_folder("目标位置 (m)")
    with position_folder:
        pos_sliders = [
            server.gui.add_slider(
                axis,
                min=float(lower[i]),
                max=float(upper[i]),
                step=0.001,
                initial_value=float(tip[i, 3]),
            )
            for i, axis in enumerate(("x", "y", "z"))
        ]

    start_rpy = np.degrees(matrix_to_rpy(tip[:3, :3]))
    orientation_folder = server.gui.add_folder("目标姿态 (deg)")
    with orientation_folder:
        rpy_sliders = [
            server.gui.add_slider(
                axis,
                min=-180.0,
                max=180.0,
                step=0.5,
                initial_value=float(start_rpy[i]),
            )
            for i, axis in enumerate(("roll", "pitch", "yaw"))
        ]
        use_orientation = server.gui.add_checkbox("约束姿态", initial_value=False)

    status = server.gui.add_text("状态", initial_value="idle")
    reset = server.gui.add_button("回到中位姿")
    _add_workspace_toggle(server, robot)

    # viser dispatches callbacks on a thread pool, so two drags can overlap.
    # The lock serialises them; it is reentrant because writing a slider's value
    # fires that slider's callback synchronously on the same thread.
    lock = threading.RLock()
    # Writing to a slider fires its callback, so guard against the two controls
    # re-triggering each other.
    syncing = {"active": False}
    real_controls: _RealArmControls | None = None

    def render(q: FloatArray) -> None:
        model.update_cfg(q)
        _update_frame(reached_frame, robot.fk(q))

    def solve_and_render(target: FloatArray, constrain_rotation: bool) -> None:
        result = (
            robot.ik(target=target)
            if constrain_rotation
            else robot.ik(position=target[:3, 3])
        )
        render(result.q)

        detail = (
            f"pos={result.position_error * 1000:.1f}mm "
            f"rot={np.degrees(result.orientation_error):.1f}deg"
        )
        if result.status.is_usable:
            status.value = f"{result.status.value}  {detail}"
        elif not constrain_rotation:
            status.value = f"该位置不可达 ({result.status.value})  {detail}"
        else:
            # Re-check without the orientation constraint to tell the user which
            # half of the request is the problem.
            position_only = robot.ik(position=target[:3, 3])
            if position_only.status.is_usable:
                status.value = f"位置可达，但该姿态不可达  {detail}"
            else:
                status.value = f"位置与姿态都不可达 ({result.status.value})  {detail}"

        # An unusable result is still rendered as the nearest pose, but it must
        # never become a real-arm command.  The CDS bus write is deliberately
        # non-blocking with respect to servo arrival; the servos enforce the
        # configured speed while moving toward the latest goal.
        commandable = result.status is IKStatus.CONVERGED or (
            not constrain_rotation and result.status is IKStatus.POSITION_ONLY
        )
        if driver is not None and commandable:
            try:
                driver.command(result.q)
            except Exception as exc:
                if real_controls is not None:
                    real_controls.report_command_error(exc)
                status.value = f"实机发送失败: {exc}"

    def target_from_sliders() -> FloatArray:
        target = np.eye(4)
        target[:3, :3] = rpy_to_matrix(np.radians([s.value for s in rpy_sliders]))
        target[:3, 3] = [s.value for s in pos_sliders]
        return target

    def sync_from_q(q: FloatArray) -> None:
        """Align target controls to a feedback pose without issuing a goal."""
        with lock:
            q = robot.clamp(q)
            pose = robot.fk(q)
            rpy = np.degrees(matrix_to_rpy(pose[:3, :3]))
            syncing["active"] = True
            try:
                for slider, value in zip(pos_sliders, pose[:3, 3], strict=True):
                    slider.value = float(value)
                for slider, value in zip(rpy_sliders, rpy, strict=True):
                    slider.value = float(np.clip(value, -180.0, 180.0))
                gizmo.position = pose[:3, 3]
                gizmo.wxyz = tf.SO3.from_matrix(pose[:3, :3]).wxyz
            finally:
                syncing["active"] = False
            render(q)
            status.value = "已同步实机当前位置"

    def on_slider_change(_args: object = None) -> None:
        with lock:
            if syncing["active"]:
                return
            target = target_from_sliders()
            syncing["active"] = True
            try:
                gizmo.position = target[:3, 3]
                gizmo.wxyz = tf.SO3.from_matrix(target[:3, :3]).wxyz
            finally:
                syncing["active"] = False
            solve_and_render(target, bool(use_orientation.value))

    def on_gizmo_change(_args: object = None) -> None:
        with lock:
            if syncing["active"]:
                return
            target = np.eye(4)
            target[:3, :3] = tf.SO3(gizmo.wxyz).as_matrix()
            target[:3, 3] = np.asarray(gizmo.position, dtype=float)
            rpy = np.degrees(matrix_to_rpy(target[:3, :3]))
            syncing["active"] = True
            try:
                for slider, value in zip(pos_sliders, target[:3, 3], strict=True):
                    # Dragging can leave the reachable box the sliders span.
                    slider.value = float(np.clip(value, slider.min, slider.max))
                for slider, value in zip(rpy_sliders, rpy, strict=True):
                    slider.value = float(np.clip(value, -180.0, 180.0))
            finally:
                syncing["active"] = False
            solve_and_render(target, bool(use_orientation.value))

    def on_reset(_args: object = None) -> None:
        with lock:
            home = robot.fk(robot.mid_range)
            home_rpy = np.degrees(matrix_to_rpy(home[:3, :3]))
            syncing["active"] = True
            try:
                for slider, value in zip(pos_sliders, home[:3, 3], strict=True):
                    slider.value = float(value)
                for slider, value in zip(rpy_sliders, home_rpy, strict=True):
                    slider.value = float(value)
                gizmo.position = home[:3, 3]
                gizmo.wxyz = tf.SO3.from_matrix(home[:3, :3]).wxyz
            finally:
                syncing["active"] = False
            solve_and_render(home, bool(use_orientation.value))

    for slider in (*pos_sliders, *rpy_sliders):
        slider.on_update(on_slider_change)
    use_orientation.on_update(on_slider_change)
    gizmo.on_update(on_gizmo_change)
    reset.on_click(on_reset)

    real_controls = _RealArmControls(server, driver, lock, sync_from_q)
    on_slider_change()
    logger.info("viser IK app 启动: http://%s:%d", host, port)
    while True:
        time.sleep(1.0)


def replay(
    robot: RobotModel,
    servo_backend: ServoBackend,
    period: float = 0.05,
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Animate the digital twin from real servo positions.

    Reads tick positions, converts to joint radians, and re-poses the model.
    No IK is involved, so this is the fastest way to catch a wrong
    ``center_tick`` or ``direction``: if the hardware pose and the browser pose
    disagree, the calibration file is the likely culprit.
    """
    from ..servo import ServoMap

    cal_path = Path(__file__).resolve().parents[1] / "config" / "servo_calibration.yaml"
    servo_map = ServoMap.from_yaml(cal_path, robot.joint_names)

    server, model = _load_viser_model(robot, host=host, port=port)
    tip_frame = _add_tip_frame(server, "/tip", robot.fk(robot.mid_range))
    _add_workspace_toggle(server, robot)

    logger.info("viser replay 启动: http://%s:%d，正在读取舵机…", host, port)
    while True:
        ticks = servo_backend.read_positions()
        q = servo_map.to_joints(ticks)
        model.update_cfg(q)
        _update_frame(tip_frame, robot.fk(q))
        time.sleep(period)
