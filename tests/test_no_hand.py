"""Tests for the no-hand check and the live test's display of it."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "capture")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import live_test  # noqa: E402
from common.no_hand import DEFAULT_THRESHOLD, EmptySquareCheck  # noqa: E402


def empty_square(brightness=0):
    rng = np.random.default_rng(0)
    image = np.full((128, 128, 3), (200, 205, 210), np.uint8)
    image[:6], image[-6:], image[:, :6], image[:, -6:] = (90, 60, 200), (90, 60, 200), (90, 60, 200), (90, 60, 200)
    noise = rng.normal(0, 3, image.shape)
    return np.clip(image.astype(np.float32) + noise + brightness, 0, 255).astype(np.uint8)


SKIN = (120, 150, 200)


def with_finger(image):
    """One extended finger plus the back of the hand entering from the left, as in the real 1-finger images."""
    image = image.copy()
    cv2.rectangle(image, (0, 40), (40, 100), SKIN, -1)   # back of the hand
    cv2.rectangle(image, (40, 60), (100, 68), SKIN, -1)  # one finger
    return image


class EmptySquareCheckTests(unittest.TestCase):
    def setUp(self):
        self.check = EmptySquareCheck()
        self.check.set_background(empty_square())

    def test_no_snapshot_never_says_empty(self):
        self.assertEqual(EmptySquareCheck().check(empty_square()), (None, False))

    def test_same_empty_square_is_empty(self):
        diff, empty = self.check.check(empty_square())
        self.assertLess(diff, 0.5)
        self.assertTrue(empty)

    def test_brightness_change_alone_is_still_empty(self):
        for shift in (-25, 15):
            diff, empty = self.check.check(empty_square(shift))
            self.assertTrue(empty, f"brightness {shift:+} gave difference {diff:.1f}")

    def test_single_finger_is_not_empty(self):
        diff, empty = self.check.check(with_finger(empty_square()))
        self.assertFalse(empty)
        self.assertGreater(diff, DEFAULT_THRESHOLD)

    def test_finger_under_brightness_change_is_not_empty(self):
        self.assertFalse(self.check.check(with_finger(empty_square(15)))[1])

    def test_lone_fingertip_is_below_threshold(self):
        # Known limit: the mean is over the whole square, so something ~3% of its area stays under it.
        image = empty_square()
        cv2.rectangle(image, (0, 60), (70, 68), SKIN, -1)
        diff, empty = self.check.check(image)
        self.assertLess(diff, DEFAULT_THRESHOLD)
        self.assertTrue(empty)

    def test_threshold_is_used(self):
        diff, _ = self.check.check(with_finger(empty_square()))
        self.check.threshold = diff + 1
        self.assertTrue(self.check.check(with_finger(empty_square()))[1])


class DisplayTests(unittest.TestCase):
    def test_empty_overrides_the_model_on_screen(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        probs = np.array([0.0, 0.0, 0.9, 0.1, 0.0, 0.0])
        for diff, empty in ((None, False), (1.2, True), (20.0, False)):
            screen = live_test.draw(frame, (10, 10, 100, 100), probs, 0.6, 3, diff, empty, 6.0)
            self.assertEqual(screen.shape, (480, 640 + live_test.PANEL_WIDTH, 3))
        # The ROI box is drawn in the "no hand" color only when the check overrides.
        empty = live_test.draw(frame, (10, 10, 100, 100), probs, 0.6, 3, 1.2, True, 6.0)
        model = live_test.draw(frame, (10, 10, 100, 100), probs, 0.6, 3, 20.0, False, 6.0)
        self.assertEqual(tuple(empty[10, 50]), live_test.CYAN)
        self.assertEqual(tuple(model[10, 50]), live_test.GREEN)


if __name__ == "__main__":
    unittest.main()
