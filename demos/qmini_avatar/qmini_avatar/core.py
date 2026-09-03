"""Pure mapping, planning, and latest-target execution for the Qmini Avatar demo.

The camera/UI entry point lives in :mod:`qmini_avatar.app`.  This module keeps
MediaPipe and OpenCV out of the control logic so the safety-critical pieces can
be exercised without a camera or robot attached.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from arm_ik import RobotModel
from arm_ik.collision import CollisionChecker
from arm_ik.servo import ServoMap
from arm_ik.workspace import sample_workspace

FloatArray = npt.NDArray[np.float64]


class PlanningError(RuntimeError):
    """A requested Avatar pose failed a pre-command safety check."""


class ArmCommandBackend(Protocol):
    """Writable arm operations used by the realtime worker."""

    def send_goals(
        self,
        positions: dict[int, int],
        speed: int,
        *,
        verify: bool = True,
    ) -> bytes: ...


@dataclass(frozen=True)
class SerialPortInfo:
    """Portable subset of ``serial.tools.list_ports.ListPortInfo``."""

    device: str
    description: str = ""
    hwid: str = ""

    @property
    def searchable(self) -> str:
        return f"{self.device} {self.description} {self.hwid}".lower()


@dataclass(frozen=True)
class SerialRoleScores:
    port: SerialPortInfo
    arm: int
    hand: int


def score_avatar_serial_ports(
    ports: Sequence[SerialPortInfo],
) -> list[SerialRoleScores]:
    """Score serial devices for the two physically different USB links.

    The Qmini arm normally uses an adapter-style link (UP-Debugger, CH340,
    CP210x, FTDI, ``usbserial``/``ttyUSB``), while the five-finger UNO direct
    link normally appears as Arduino, ``usbmodem``, or ``ttyACM``.  Scores are
    intentionally conservative so an ambiguous pair is reported, not guessed.
    """

    result: list[SerialRoleScores] = []
    for port in ports:
        text = port.searchable
        arm = 0
        hand = 0
        if "up-debugger" in text or "up debugger" in text:
            arm += 160
        if "usbserial" in text:
            arm += 130
        if "ttyusb" in text:
            arm += 120
        if any(token in text for token in ("ch340", "ch341", "wchusbserial")):
            arm += 100
        if any(token in text for token in ("cp210", "ftdi")):
            arm += 90

        if "usbmodem" in text:
            hand += 140
        if "ttyacm" in text:
            hand += 130
        if "arduino" in text or "genuino" in text:
            hand += 120
        if "uno" in text:
            hand += 80

        # Strong role evidence should suppress a weak cross-role clue.  This is
        # especially important for CH340-based Arduino clones on Windows.
        if ("arduino" in text or "uno" in text) and hand > 0:
            arm = max(0, arm - 100)
        if any(token in text for token in ("usbserial", "ttyusb", "up-debugger")):
            hand = max(0, hand - 100)
        result.append(SerialRoleScores(port, arm, hand))
    return result


def choose_avatar_serial_ports(
    ports: Sequence[SerialPortInfo],
    *,
    want_arm: bool,
    want_hand: bool,
    arm_override: str | None = None,
    hand_override: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve requested arm/hand ports without ever sharing one device."""

    arm_explicit = arm_override not in (None, "auto")
    hand_explicit = hand_override not in (None, "auto")
    arm = arm_override if arm_explicit else None
    hand = hand_override if hand_explicit else None
    if arm is not None and hand is not None and arm == hand:
        raise RuntimeError("arm and uHand cannot use the same serial port")

    scores = score_avatar_serial_ports(ports)

    def pick(role: str, excluded: set[str]) -> str:
        ranked = sorted(
            (
                (getattr(item, role), item.port.device)
                for item in scores
                if item.port.device not in excluded and getattr(item, role) > 0
            ),
            reverse=True,
        )
        if not ranked:
            inventory = (
                ", ".join(item.port.device for item in scores) or "no serial ports"
            )
            raise RuntimeError(f"could not identify {role} port; found: {inventory}")
        top_score = ranked[0][0]
        tied = sorted(device for score, device in ranked if score == top_score)
        if len(tied) != 1:
            raise RuntimeError(f"ambiguous {role} ports {tied}; pass --{role}-port")
        return tied[0]

    if want_arm and arm is None:
        arm = pick("arm", {hand} if hand is not None else set())
    if want_hand and hand is None:
        hand = pick("hand", {arm} if arm is not None else set())
    if arm is not None and hand is not None and arm == hand:
        raise RuntimeError("automatic detection assigned one port twice")
    return arm if want_arm else None, hand if want_hand else None


@dataclass(frozen=True)
class HumanHandPose:
    """Camera-space features needed by the relative Avatar mapping."""

    palm_x: float
    palm_y: float
    palm_scale: float
    roll: float
    closures: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class FingerVisionCalibration:
    """Per-user visual finger scores plus logical uHand angle endpoints."""

    closed_scores: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    open_scores: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    closed_angles: tuple[int, ...] = (0, 0, 0, 0, 0)
    open_angles: tuple[int, ...] = (180, 180, 180, 180, 180)

    def __post_init__(self) -> None:
        for name in ("closed_scores", "open_scores", "closed_angles", "open_angles"):
            values = getattr(self, name)
            if len(values) != 5:
                raise ValueError(f"{name} must contain five values")
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} must contain finite values")
        if any(
            not 0 <= int(value) <= 180
            for value in self.closed_angles + self.open_angles
        ):
            raise ValueError("finger endpoint angles must be in 0..180")

    @classmethod
    def load(cls, path: Path) -> "FingerVisionCalibration":
        """Load a five-finger visual calibration JSON file."""

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"could not read finger calibration {path}: {exc}"
            ) from exc
        defaults = cls()

        def five(
            name: str, fallback: tuple[float, ...] | tuple[int, ...]
        ) -> tuple[object, ...]:
            values = raw.get(name, fallback)
            if not isinstance(values, list) or len(values) < 5:
                return tuple(fallback)
            return tuple(values[:5])

        return cls(
            closed_scores=tuple(
                float(v) for v in five("closed_scores", defaults.closed_scores)
            ),
            open_scores=tuple(
                float(v) for v in five("open_scores", defaults.open_scores)
            ),
            closed_angles=tuple(
                int(v) for v in five("closed_angles", defaults.closed_angles)
            ),
            open_angles=tuple(
                int(v) for v in five("open_angles", defaults.open_angles)
            ),
        )

    def closures_for_scores(self, scores: Sequence[float]) -> tuple[float, ...]:
        """Convert hand-mirror open scores to normalized closure values."""

        if len(scores) != 5:
            raise ValueError("finger scores must contain five values")
        closures: list[float] = []
        for score, closed, opened in zip(scores, self.closed_scores, self.open_scores):
            if abs(opened - closed) < 1e-5:
                openness = float(np.clip(score, 0.0, 1.0))
            else:
                openness = float(
                    np.clip((score - closed) / (opened - closed), 0.0, 1.0)
                )
            closures.append(1.0 - openness)
        return tuple(closures)

    def angles_for_closures(self, closures: Sequence[float]) -> tuple[int, ...]:
        """Apply the same calibrated logical servo endpoints as hand mirror."""

        if len(closures) != 5:
            raise ValueError("finger closures must contain five values")
        return tuple(
            int(round(np.clip(opened + float(closure) * (closed - opened), 0.0, 180.0)))
            for closure, closed, opened in zip(
                closures, self.closed_angles, self.open_angles
            )
        )


@dataclass(frozen=True)
class AvatarTarget:
    """One arm-and-hand target in robot coordinates."""

    position: FloatArray
    servo6: float
    closures: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class WorkspaceProjection:
    """A desired position reconciled with a sampled reachable workspace."""

    position: FloatArray
    sample_position: FloatArray
    sample_q: FloatArray
    distance: float
    projected: bool


class ReachableWorkspaceProjector:
    """Project Cartesian targets onto sampled collision-free arm positions.

    A small coverage tolerance preserves smooth Cartesian motion inside the
    sampled shell.  Targets farther from every known-reachable sample snap to a
    nearby sample, which prevents a rectangular hand-control volume from
    repeatedly asking IK for points inside the arm's unreachable hollow core.
    """

    def __init__(
        self,
        robot: RobotModel,
        q: npt.ArrayLike,
        positions: npt.ArrayLike,
        *,
        coverage_tolerance_m: float = 0.012,
        continuity_weight_m: float = 0.004,
        neighbours: int = 48,
    ) -> None:
        q_array = np.asarray(q, dtype=np.float64).reshape(-1, robot.dof)
        position_array = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if len(q_array) != len(position_array) or len(q_array) < 10:
            raise ValueError("workspace needs matching q/position samples")
        if coverage_tolerance_m <= 0 or continuity_weight_m < 0 or neighbours < 1:
            raise ValueError("workspace projection settings are invalid")
        self.robot = robot
        self.q = q_array
        self.positions = position_array
        self.coverage_tolerance_m = coverage_tolerance_m
        self.continuity_weight_m = continuity_weight_m
        self.neighbours = min(neighbours, len(q_array))
        self._span = np.maximum(robot.upper - robot.lower, 1e-9)

    @classmethod
    def sample(
        cls,
        robot: RobotModel,
        collision_checker: CollisionChecker,
        *,
        count: int = 6000,
        seed: int = 2026,
        coverage_tolerance_m: float = 0.012,
    ) -> ReachableWorkspaceProjector:
        if count < 100:
            raise ValueError("workspace sample count must be at least 100")
        workspace = sample_workspace(robot, count=count - 1, seed=seed)
        configurations: list[FloatArray] = [robot.mid_range.copy()]
        positions: list[FloatArray] = [robot.fk(robot.mid_range)[:3, 3].copy()]
        for q, position in zip(workspace.q, workspace.positions, strict=True):
            if collision_checker.is_free(q):
                configurations.append(q)
                positions.append(position.copy())
        return cls(
            robot,
            np.asarray(configurations),
            np.asarray(positions),
            coverage_tolerance_m=coverage_tolerance_m,
        )

    @property
    def size(self) -> int:
        return len(self.q)

    @property
    def bounds(self) -> tuple[FloatArray, FloatArray]:
        return self.positions.min(axis=0), self.positions.max(axis=0)

    def project(
        self, desired: npt.ArrayLike, current_q: npt.ArrayLike
    ) -> WorkspaceProjection:
        position = np.asarray(desired, dtype=np.float64).reshape(3)
        current = np.asarray(current_q, dtype=np.float64).reshape(self.robot.dof)
        squared = np.sum((self.positions - position[None, :]) ** 2, axis=1)
        if self.neighbours == len(squared):
            candidates = np.arange(len(squared))
        else:
            candidates = np.argpartition(squared, self.neighbours - 1)[
                : self.neighbours
            ]
        distances = np.sqrt(squared[candidates])
        joint_distance = np.max(
            np.abs((self.q[candidates] - current[None, :]) / self._span[None, :]),
            axis=1,
        )
        score = distances + self.continuity_weight_m * joint_distance
        chosen = int(candidates[int(np.argmin(score))])
        nearest_distance = float(np.sqrt(float(squared.min())))
        sample_position = self.positions[chosen].copy()
        projected = nearest_distance > self.coverage_tolerance_m
        return WorkspaceProjection(
            position=sample_position if projected else position.copy(),
            sample_position=sample_position,
            sample_q=self.q[chosen].copy(),
            distance=float(np.linalg.norm(sample_position - position)),
            projected=projected,
        )


@dataclass(frozen=True)
class AvatarMappingConfig:
    """Gains and hard relative-motion bounds for monocular control."""

    depth_gain_m: float = 0.14
    lateral_gain_m: float = 0.25
    vertical_gain_m: float = 0.25
    roll_gain: float = 1.0
    max_depth_m: float = 0.055
    max_lateral_m: float = 0.075
    max_vertical_m: float = 0.075

    def __post_init__(self) -> None:
        values = (
            self.depth_gain_m,
            self.lateral_gain_m,
            self.vertical_gain_m,
            self.max_depth_m,
            self.max_lateral_m,
            self.max_vertical_m,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError(
                "Avatar mapping gains and bounds must be finite and positive"
            )
        if not math.isfinite(self.roll_gain) or self.roll_gain == 0:
            raise ValueError("roll_gain must be finite and non-zero")


@dataclass(frozen=True)
class AvatarCalibration:
    human: HumanHandPose
    arm_position: FloatArray
    servo6: float


def _point_xyz(point: object) -> FloatArray:
    """Read MediaPipe-like x/y/z attributes without importing MediaPipe."""

    try:
        return np.array(
            [
                float(getattr(point, "x")),
                float(getattr(point, "y")),
                float(getattr(point, "z", 0.0)),
            ],
            dtype=np.float64,
        )
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("landmark must expose finite numeric x/y/z values") from exc


def joint_angle(a: object, b: object, c: object) -> float:
    """Return angle ABC in radians for three MediaPipe-like landmarks."""

    pa, pb, pc = _point_xyz(a), _point_xyz(b), _point_xyz(c)
    first, second = pa - pb, pc - pb
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom < 1e-12:
        return math.pi
    cosine = float(np.clip(np.dot(first, second) / denom, -1.0, 1.0))
    return math.acos(cosine)


_FINGER_LANDMARKS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
_PALM_LANDMARKS = (0, 5, 9, 13, 17)


def extract_hand_pose(
    image_landmarks: Sequence[object],
    flex_landmarks: Sequence[object] | None = None,
    *,
    finger_calibration: FingerVisionCalibration | None = None,
) -> HumanHandPose:
    """Extract relative-control features from 21 hand landmarks.

    Image landmarks provide stable screen position, apparent scale, and roll.
    MediaPipe world landmarks may optionally provide less perspective-sensitive
    finger flexion angles.
    """

    if len(image_landmarks) != 21:
        raise ValueError(f"expected 21 image landmarks, got {len(image_landmarks)}")
    flex = image_landmarks if flex_landmarks is None else flex_landmarks
    if len(flex) != 21:
        raise ValueError(f"expected 21 flex landmarks, got {len(flex)}")
    calibration = finger_calibration or FingerVisionCalibration()

    image = np.array([_point_xyz(point) for point in image_landmarks])
    if not np.all(np.isfinite(image)):
        raise ValueError("hand landmarks must be finite")
    centre = image[list(_PALM_LANDMARKS), :2].mean(axis=0)
    palm_width = float(np.linalg.norm(image[5, :2] - image[17, :2]))
    palm_length = float(np.linalg.norm(image[0, :2] - image[9, :2]))
    scale = 0.5 * (palm_width + palm_length)
    if scale < 1e-4:
        raise ValueError("detected palm is too small for stable control")

    edge = image[17, :2] - image[5, :2]
    roll = math.atan2(float(edge[1]), float(edge[0]))
    scores: list[float] = []
    for mcp, pip, dip, tip in _FINGER_LANDMARKS:
        bend = joint_angle(flex[mcp], flex[pip], flex[dip])
        bend += joint_angle(flex[pip], flex[dip], flex[tip])
        # An open score is the two included joint angles divided by 360 degrees.
        scores.append(float(np.clip(bend / (2.0 * math.pi), 0.0, 1.0)))
    closures = calibration.closures_for_scores(scores)

    return HumanHandPose(
        palm_x=float(centre[0]),
        palm_y=float(centre[1]),
        palm_scale=scale,
        roll=roll,
        closures=closures,  # type: ignore[arg-type]
    )


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""

    return math.atan2(math.sin(angle), math.cos(angle))


class AvatarMapper:
    """Map a human hand around a captured neutral pose into robot targets."""

    def __init__(
        self,
        servo6_limits: tuple[float, float],
        config: AvatarMappingConfig | None = None,
    ) -> None:
        self.config = config or AvatarMappingConfig()
        self.servo6_limits = (float(servo6_limits[0]), float(servo6_limits[1]))
        self.calibration: AvatarCalibration | None = None

    @property
    def calibrated(self) -> bool:
        return self.calibration is not None

    def calibrate(
        self,
        human: HumanHandPose,
        arm_position: npt.ArrayLike,
        servo6: float,
    ) -> AvatarCalibration:
        position = np.asarray(arm_position, dtype=np.float64).reshape(3).copy()
        if not np.all(np.isfinite(position)) or human.palm_scale <= 0:
            raise ValueError("calibration values must be finite")
        low, high = self.servo6_limits
        if not low <= float(servo6) <= high:
            raise ValueError("servo6 calibration angle is outside its limits")
        self.calibration = AvatarCalibration(human, position, float(servo6))
        return self.calibration

    def map(self, human: HumanHandPose) -> AvatarTarget:
        if self.calibration is None:
            raise RuntimeError("press C to capture a neutral pose before mapping")
        base = self.calibration
        cfg = self.config
        if human.palm_scale <= 0:
            raise ValueError("palm scale must be positive")

        # A larger apparent palm means the hand moved towards the camera.  The
        # log ratio makes towards/away motion symmetric around calibration.
        depth = cfg.depth_gain_m * math.log(human.palm_scale / base.human.palm_scale)
        lateral = -cfg.lateral_gain_m * (human.palm_x - base.human.palm_x)
        vertical = -cfg.vertical_gain_m * (human.palm_y - base.human.palm_y)
        offset = np.array(
            [
                np.clip(depth, -cfg.max_depth_m, cfg.max_depth_m),
                np.clip(lateral, -cfg.max_lateral_m, cfg.max_lateral_m),
                np.clip(vertical, -cfg.max_vertical_m, cfg.max_vertical_m),
            ],
            dtype=np.float64,
        )
        roll_delta = cfg.roll_gain * wrap_angle(human.roll - base.human.roll)
        servo6 = float(
            np.clip(
                base.servo6 + roll_delta,
                self.servo6_limits[0],
                self.servo6_limits[1],
            )
        )
        return AvatarTarget(
            position=base.arm_position + offset,
            servo6=servo6,
            closures=human.closures,
        )


class TargetSmoother:
    """EMA smoothing plus per-update slew limits for arm, wrist, and fingers."""

    def __init__(
        self,
        *,
        alpha: float = 0.45,
        max_position_step_m: float = 0.006,
        max_servo6_step_deg: float = 4.0,
        max_closure_step: float = 0.12,
    ) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if min(max_position_step_m, max_servo6_step_deg, max_closure_step) <= 0:
            raise ValueError("smoothing step limits must be positive")
        self.alpha = alpha
        self.max_position_step_m = max_position_step_m
        self.max_servo6_step = math.radians(max_servo6_step_deg)
        self.max_closure_step = max_closure_step
        self._value: AvatarTarget | None = None

    def reset(self, value: AvatarTarget | None = None) -> None:
        self._value = value

    def update(self, target: AvatarTarget) -> AvatarTarget:
        if self._value is None:
            self._value = target
            return target
        old = self._value
        desired_position = old.position + self.alpha * (target.position - old.position)
        position_delta = np.clip(
            desired_position - old.position,
            -self.max_position_step_m,
            self.max_position_step_m,
        )
        desired_servo6 = old.servo6 + self.alpha * (target.servo6 - old.servo6)
        servo6_delta = float(
            np.clip(
                desired_servo6 - old.servo6, -self.max_servo6_step, self.max_servo6_step
            )
        )
        old_closures = np.asarray(old.closures)
        desired_closures = old_closures + self.alpha * (
            np.asarray(target.closures) - old_closures
        )
        closures = old_closures + np.clip(
            desired_closures - old_closures,
            -self.max_closure_step,
            self.max_closure_step,
        )
        self._value = AvatarTarget(
            position=old.position + position_delta,
            servo6=old.servo6 + servo6_delta,
            closures=tuple(float(value) for value in closures),  # type: ignore[arg-type]
        )
        return self._value


class FingerCommandFilter:
    """Calibrated angle-domain EMA and per-command slew limiting."""

    def __init__(
        self,
        calibration: FingerVisionCalibration,
        *,
        alpha: float = 0.65,
        max_step_deg: float = 12.0,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("finger smoothing must be in (0, 1]")
        if not math.isfinite(max_step_deg) or max_step_deg <= 0:
            raise ValueError("finger max step must be positive")
        self.calibration = calibration
        self.alpha = alpha
        self.max_step_deg = max_step_deg
        self._angles = np.asarray(calibration.open_angles, dtype=np.float64)

    def reset_open(self) -> tuple[int, ...]:
        self._angles = np.asarray(self.calibration.open_angles, dtype=np.float64)
        return tuple(int(value) for value in self.calibration.open_angles)

    def update(self, closures: Sequence[float]) -> tuple[int, ...]:
        target = np.asarray(
            self.calibration.angles_for_closures(closures), dtype=np.float64
        )
        ema = self._angles + self.alpha * (target - self._angles)
        self._angles += np.clip(
            ema - self._angles,
            -self.max_step_deg,
            self.max_step_deg,
        )
        return tuple(int(round(np.clip(value, 0.0, 180.0))) for value in self._angles)


@dataclass(frozen=True)
class PlannedAvatarMove:
    target: AvatarTarget
    q: FloatArray
    ticks: dict[int, int]
    position_error: float
    max_joint_step_deg: float
    projection_distance: float = 0.0
    projected: bool = False
    slew_limited: bool = False


class AvatarPlanner:
    """Position IK with servo6 fixing and sampled joint-path collision checks."""

    def __init__(
        self,
        robot: RobotModel,
        servo_map: ServoMap,
        *,
        collision_checker: CollisionChecker | None = None,
        workspace: ReachableWorkspaceProjector | None = None,
        max_joint_step_deg: float = 10.0,
        collision_samples: int = 10,
    ) -> None:
        if max_joint_step_deg <= 0 or collision_samples < 2:
            raise ValueError("planner safety limits are invalid")
        self.robot = robot
        self.servo_map = servo_map
        self.collision_checker = collision_checker or CollisionChecker(robot)
        self.workspace = workspace
        self.max_joint_step_deg = max_joint_step_deg
        self.collision_samples = collision_samples

    def plan(self, current_q: npt.ArrayLike, target: AvatarTarget) -> PlannedAvatarMove:
        current = np.asarray(current_q, dtype=np.float64).reshape(self.robot.dof)
        clamped = self.robot.clamp(current)
        if float(np.max(np.abs(np.degrees(clamped - current)))) > 0.5:
            raise PlanningError("arm feedback is outside the effective joint limits")
        current = clamped
        projection = None
        planned_target = target
        if self.workspace is not None:
            projection = self.workspace.project(target.position, current)
            planned_target = AvatarTarget(
                position=projection.position,
                servo6=target.servo6,
                closures=target.closures,
            )

        result = self.robot.ik_position(
            planned_target.position,
            servo6=target.servo6,
            seed=current,
        )
        # A nearby sampled configuration proves that its exact position is
        # reachable.  Use it as a fallback when the current IK branch is pinned
        # at a limit, and snap to that sample if a tolerance-preserved point was
        # still too optimistic for the thin shell.
        if not result.status.is_usable and projection is not None:
            planned_target = AvatarTarget(
                position=projection.sample_position,
                servo6=target.servo6,
                closures=target.closures,
            )
            result = self.robot.ik_position(
                planned_target.position,
                servo6=target.servo6,
                seed=projection.sample_q,
            )
            projection = replace(
                projection, position=projection.sample_position, projected=True
            )
        if not result.status.is_usable:
            raise PlanningError(
                f"IK {result.status.value}, error={result.position_error * 1000:.1f} mm"
            )

        requested_delta = float(np.max(np.abs(np.degrees(result.q - current))))
        slew_limited = requested_delta > self.max_joint_step_deg
        command_q = result.q.copy()
        if slew_limited:
            # A projected sample can be on another IK branch.  Rejecting it
            # forever leaves the Avatar stuck at the workspace boundary. Move a
            # bounded fraction through joint space instead: every intermediate
            # configuration remains within limits and is collision-checked
            # below before it can be commanded.
            fraction = self.max_joint_step_deg / requested_delta
            command_q = current + fraction * (result.q - current)
        max_delta = float(np.max(np.abs(np.degrees(command_q - current))))

        ticks = self.servo_map.to_ticks(command_q)
        tick_q = self.servo_map.to_joints(ticks)
        quantization = float(np.max(np.abs(np.degrees(tick_q - command_q))))
        if quantization > 0.5:
            raise PlanningError(
                f"servo mapping changed a joint by {quantization:.2f} deg"
            )

        for index, fraction in enumerate(np.linspace(0.0, 1.0, self.collision_samples)):
            q = current * (1.0 - fraction) + command_q * fraction
            collisions = self.collision_checker.check(q)
            if collisions:
                detail = ", ".join(str(pair) for pair in collisions[:2])
                raise PlanningError(
                    f"self collision at path sample {index + 1}/{self.collision_samples}: {detail}"
                )

        return PlannedAvatarMove(
            target=planned_target,
            q=command_q.copy(),
            ticks=ticks,
            position_error=float(
                np.linalg.norm(
                    self.robot.fk(command_q)[:3, 3] - planned_target.position
                )
            ),
            max_joint_step_deg=max_delta,
            projection_distance=(0.0 if projection is None else projection.distance),
            projected=(False if projection is None else projection.projected),
            slew_limited=slew_limited,
        )


@dataclass(frozen=True)
class MotionState:
    status: str
    message: str
    q_command: FloatArray
    last_target: AvatarTarget | None = None
    sent_count: int = 0
    updated_at: float = 0.0


class AvatarMotionWorker:
    """Plan and send only the newest arm target at a bounded frequency."""

    def __init__(
        self,
        planner: AvatarPlanner,
        initial_q: npt.ArrayLike,
        *,
        backend: ArmCommandBackend | None = None,
        speed: int = 120,
        rate_hz: float = 10.0,
    ) -> None:
        if not 1 <= speed <= 1023 or rate_hz <= 0:
            raise ValueError("arm speed/rate is invalid")
        self.planner = planner
        self.backend = backend
        self.speed = speed
        self.period = 1.0 / rate_hz
        q = planner.robot.clamp(initial_q)
        self._condition = threading.Condition()
        self._pending: AvatarTarget | None = None
        self._enabled = False
        self._failed = False
        self._stopping = False
        self._state = MotionState("paused", "press SPACE to start", q.copy())
        self._thread = threading.Thread(
            target=self._run,
            name="qmini-avatar-arm",
            daemon=True,
        )
        self._thread.start()

    def enable(self) -> None:
        with self._condition:
            if self._stopping:
                raise RuntimeError("motion worker is closed")
            if self._failed:
                raise RuntimeError("motion worker failed; restart the program")
            self._enabled = True
            self._state = replace(
                self._state,
                status="ready",
                message="waiting for hand target",
                updated_at=time.monotonic(),
            )
            self._condition.notify_all()

    def pause(self, message: str = "paused") -> None:
        with self._condition:
            self._enabled = False
            self._pending = None
            self._state = replace(
                self._state,
                status="paused",
                message=message,
                updated_at=time.monotonic(),
            )
            self._condition.notify_all()

    def submit(self, target: AvatarTarget) -> None:
        with self._condition:
            if self._enabled and not self._stopping:
                self._pending = target
                self._condition.notify_all()

    def snapshot(self) -> MotionState:
        with self._condition:
            state = self._state
            return replace(state, q_command=state.q_command.copy())

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._enabled = False
            self._pending = None
            self._condition.notify_all()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        next_allowed = time.monotonic()
        while True:
            with self._condition:
                while not self._stopping and (
                    not self._enabled or self._pending is None
                ):
                    self._condition.wait(timeout=0.25)
                if self._stopping:
                    return
                remaining = next_allowed - time.monotonic()
                if remaining > 0:
                    self._condition.wait(timeout=remaining)
                    continue
                target = self._pending
                self._pending = None
                current = self._state.q_command.copy()
            assert target is not None

            try:
                planned = self.planner.plan(current, target)
                if self.backend is not None:
                    self.backend.send_goals(planned.ticks, self.speed, verify=False)
                with self._condition:
                    self._state = MotionState(
                        status="live" if self.backend is not None else "sim",
                        message=(
                            f"IK {planned.position_error * 1000:.2f} mm, "
                            f"joint step {planned.max_joint_step_deg:.1f} deg"
                            + (
                                f", workspace snap {planned.projection_distance * 1000:.1f} mm"
                                if planned.projected
                                else ""
                            )
                            + (", joint slew limited" if planned.slew_limited else "")
                        ),
                        q_command=planned.q.copy(),
                        last_target=target,
                        sent_count=self._state.sent_count + 1,
                        updated_at=time.monotonic(),
                    )
            except PlanningError as exc:
                with self._condition:
                    self._state = replace(
                        self._state,
                        status="rejected",
                        message=str(exc),
                        last_target=target,
                        updated_at=time.monotonic(),
                    )
            except Exception as exc:
                # A bus failure is terminal for this live session.  The current
                # goal remains held by the servos; no later camera frame is sent.
                with self._condition:
                    self._enabled = False
                    self._failed = True
                    self._pending = None
                    self._state = replace(
                        self._state,
                        status="error",
                        message=f"arm command failed: {exc}",
                        last_target=target,
                        updated_at=time.monotonic(),
                    )
            next_allowed = time.monotonic() + self.period


__all__ = [
    "AvatarMapper",
    "AvatarMappingConfig",
    "AvatarMotionWorker",
    "AvatarPlanner",
    "AvatarTarget",
    "FingerCommandFilter",
    "FingerVisionCalibration",
    "HumanHandPose",
    "MotionState",
    "PlannedAvatarMove",
    "PlanningError",
    "ReachableWorkspaceProjector",
    "SerialPortInfo",
    "SerialRoleScores",
    "TargetSmoother",
    "WorkspaceProjection",
    "choose_avatar_serial_ports",
    "extract_hand_pose",
    "joint_angle",
    "score_avatar_serial_ports",
    "wrap_angle",
]
