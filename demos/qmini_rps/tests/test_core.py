from __future__ import annotations

import unittest

from qmini_rps.app import build_parser, configure_capture
from qmini_rps.core import (
    Gesture,
    StableGestureRecognizer,
    classify_closures,
    winning_command,
)


class CameraConfigurationTests(unittest.TestCase):
    def test_default_profile_uses_stable_wsl_usbip_resolution(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual((args.width, args.height), (320, 240))
        self.assertEqual(args.camera_format, "mjpg")

    def test_mjpg_is_negotiated_before_other_capture_properties(self) -> None:
        class FakeCV2:
            CAP_PROP_FOURCC = 6
            CAP_PROP_BUFFERSIZE = 38
            CAP_PROP_FRAME_WIDTH = 3
            CAP_PROP_FRAME_HEIGHT = 4

            @staticmethod
            def VideoWriter_fourcc(*letters: str) -> int:
                return sum(ord(letter) << (8 * index) for index, letter in enumerate(letters))

        class FakeCapture:
            def __init__(self) -> None:
                self.calls: list[tuple[int, float]] = []

            def set(self, prop: int, value: float) -> bool:
                self.calls.append((prop, value))
                return True

        capture = FakeCapture()
        configure_capture(FakeCV2, capture, 640, 480, "mjpg")

        self.assertEqual(capture.calls[0][0], FakeCV2.CAP_PROP_FOURCC)
        self.assertEqual(
            int(capture.calls[0][1]), FakeCV2.VideoWriter_fourcc(*"MJPG")
        )
        self.assertIn((FakeCV2.CAP_PROP_FRAME_WIDTH, 640), capture.calls)
        self.assertIn((FakeCV2.CAP_PROP_FRAME_HEIGHT, 480), capture.calls)


class ClassificationTests(unittest.TestCase):
    def test_classifies_three_legal_gestures(self) -> None:
        examples = {
            Gesture.ROCK: (0.2, 0.9, 0.8, 0.85, 0.95),
            Gesture.PAPER: (0.9, 0.1, 0.2, 0.15, 0.05),
            Gesture.SCISSORS: (0.5, 0.1, 0.2, 0.8, 0.9),
        }
        for expected, closures in examples.items():
            with self.subTest(expected=expected):
                result = classify_closures(closures)
                self.assertIsNotNone(result)
                assert result is not None
                self.assertIs(result.gesture, expected)

    def test_thumb_does_not_change_rps_shape(self) -> None:
        closed_thumb = classify_closures((1.0, 0.1, 0.1, 0.9, 0.9))
        open_thumb = classify_closures((0.0, 0.1, 0.1, 0.9, 0.9))
        self.assertEqual(closed_thumb, open_thumb)

    def test_uncertain_fingers_are_not_guessed(self) -> None:
        self.assertIsNone(classify_closures((0.5, 0.5, 0.1, 0.9, 0.9)))

    def test_response_always_beats_human(self) -> None:
        expected = {
            Gesture.ROCK: (Gesture.PAPER, "open"),
            Gesture.PAPER: (Gesture.SCISSORS, "victory"),
            Gesture.SCISSORS: (Gesture.ROCK, "fist"),
        }
        for human, (robot, preset) in expected.items():
            command = winning_command(human)
            self.assertIs(command.robot, robot)
            self.assertEqual(command.uhand_gesture, preset)


class StabilityTests(unittest.TestCase):
    def test_requires_consecutive_frames(self) -> None:
        recognizer = StableGestureRecognizer(stable_frames=3)
        prediction = classify_closures((0.0, 0.9, 0.9, 0.9, 0.9))
        assert prediction is not None
        self.assertFalse(recognizer.update(prediction).changed)
        self.assertFalse(recognizer.update(prediction).changed)
        accepted = recognizer.update(prediction)
        self.assertTrue(accepted.changed)
        self.assertIs(accepted.stable, Gesture.ROCK)
        self.assertFalse(recognizer.update(prediction).changed)

    def test_uncertain_frame_resets_candidate_streak(self) -> None:
        recognizer = StableGestureRecognizer(stable_frames=2)
        prediction = classify_closures((0.0, 0.1, 0.1, 0.1, 0.1))
        assert prediction is not None
        recognizer.update(prediction)
        recognizer.update(None)
        update = recognizer.update(prediction)
        self.assertEqual(update.streak, 1)
        self.assertIsNone(update.stable)


if __name__ == "__main__":
    unittest.main()
