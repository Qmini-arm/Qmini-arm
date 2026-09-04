"""Pure rock-paper-scissors classification and response mapping."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class Gesture(str, Enum):
    """The three legal rock-paper-scissors gestures."""

    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"


@dataclass(frozen=True)
class GesturePrediction:
    gesture: Gesture
    confidence: float


@dataclass(frozen=True)
class WinningCommand:
    """A human gesture and the uHand preset that defeats it."""

    human: Gesture
    robot: Gesture
    uhand_gesture: str


_WINNING_COMMANDS = {
    Gesture.ROCK: WinningCommand(Gesture.ROCK, Gesture.PAPER, "open"),
    Gesture.PAPER: WinningCommand(Gesture.PAPER, Gesture.SCISSORS, "victory"),
    Gesture.SCISSORS: WinningCommand(Gesture.SCISSORS, Gesture.ROCK, "fist"),
}


def winning_command(human: Gesture) -> WinningCommand:
    """Return the robot gesture that beats ``human``."""

    return _WINNING_COMMANDS[Gesture(human)]


def classify_closures(
    closures: Sequence[float],
    *,
    open_threshold: float = 0.35,
    closed_threshold: float = 0.65,
) -> GesturePrediction | None:
    """Classify calibrated finger closures in thumb-to-pinky order.

    The four long fingers carry the RPS shape. Thumb closure is deliberately
    ignored because its MediaPipe curl score is much more sensitive to hand
    rotation, and a legal rock or scissors pose may place the thumb differently.
    Values between the open and closed thresholds form an uncertainty band.
    """

    if len(closures) != 5:
        raise ValueError("closures must contain thumb, index, middle, ring, pinky")
    values = tuple(float(value) for value in closures)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("closures must be finite values within [0, 1]")
    if not 0.0 <= open_threshold < closed_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= open < closed <= 1")

    _, index, middle, ring, pinky = values
    long_fingers = (index, middle, ring, pinky)

    if all(value >= closed_threshold for value in long_fingers):
        return GesturePrediction(Gesture.ROCK, min(long_fingers))
    if all(value <= open_threshold for value in long_fingers):
        return GesturePrediction(Gesture.PAPER, 1.0 - max(long_fingers))
    if (
        index <= open_threshold
        and middle <= open_threshold
        and ring >= closed_threshold
        and pinky >= closed_threshold
    ):
        confidence = min(1.0 - index, 1.0 - middle, ring, pinky)
        return GesturePrediction(Gesture.SCISSORS, confidence)
    return None


@dataclass(frozen=True)
class RecognitionUpdate:
    observed: Gesture | None
    stable: Gesture | None
    streak: int
    changed: bool
    confidence: float


class StableGestureRecognizer:
    """Require the same classification for several frames before accepting it."""

    def __init__(self, stable_frames: int = 5) -> None:
        if stable_frames < 1:
            raise ValueError("stable_frames must be at least 1")
        self.stable_frames = stable_frames
        self._candidate: Gesture | None = None
        self._streak = 0
        self._stable: Gesture | None = None

    @property
    def stable(self) -> Gesture | None:
        return self._stable

    def reset(self, *, clear_stable: bool = True) -> None:
        self._candidate = None
        self._streak = 0
        if clear_stable:
            self._stable = None

    def update(self, prediction: GesturePrediction | None) -> RecognitionUpdate:
        observed = None if prediction is None else prediction.gesture
        confidence = 0.0 if prediction is None else prediction.confidence
        if observed is None:
            self._candidate = None
            self._streak = 0
            return RecognitionUpdate(None, self._stable, 0, False, confidence)

        if observed == self._candidate:
            self._streak += 1
        else:
            self._candidate = observed
            self._streak = 1

        changed = False
        if self._streak >= self.stable_frames and observed != self._stable:
            self._stable = observed
            changed = True
        return RecognitionUpdate(observed, self._stable, self._streak, changed, confidence)
