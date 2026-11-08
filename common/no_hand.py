"""A "no hand" check that doesn't use the CNN: compare the square with a snapshot of it empty.

    check = EmptySquareCheck()
    check.set_background(crop_and_resize(frame, roi))   # when the button is pressed, square empty
    diff, empty = check.check(crop_and_resize(frame, roi))
    if empty: the answer is 0, whatever the model says

The difference is the mean absolute pixel difference (0-255) of blurred grayscale
crops, after subtracting the median difference. Subtracting the median removes a
change in overall brightness, which affects every pixel; a hand changes only part
of the square. Only OpenCV and NumPy are needed, so this also runs on the Uno Q.

Measured on the captured data (2026-11-08): an empty square against an empty
snapshot from the same session was at most 1.7, even while the room brightened
from 155 to 167; any hand was at least 17.8. A snapshot from two days earlier
differed by 11.2, so take a fresh one each time.
"""
import cv2
import numpy as np

DEFAULT_THRESHOLD = 6.0
BLUR = 5


def _gray(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(gray, (BLUR, BLUR), 0)


def difference(background_gray, crop):
    """Brightness-corrected mean absolute difference between a crop and the background."""
    change = _gray(crop) - background_gray
    return float(np.abs(change - np.median(change)).mean())


class EmptySquareCheck:
    def __init__(self, threshold=DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.background = None

    def set_background(self, crop):
        """Snapshot of the empty square (a BGR crop from crop_and_resize)."""
        self.background = _gray(crop)

    def check(self, crop):
        """Returns (difference, is_empty). Before set_background: (None, False)."""
        if self.background is None:
            return None, False
        diff = difference(self.background, crop)
        return diff, diff < self.threshold
