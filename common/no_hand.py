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

Second test, new edges (added 2026-11-09): the share of pixels with a strong edge that
the snapshot didn't have near the same place. Paper, the square's border and the red
frame are in the snapshot too, so they cancel; light, colour or shadow drift is soft
and adds no edges; a hand adds many. On AirFour the difference drifted to 6.6-10.4 on
an empty square while unlocked, so the empty square (which the CNN calls 0) counted as
a fist and relocked the lock. Measured on the captured data: empty squares, also with
simulated light, colour and shadow drift (difference up to 20.9), had 0.00% new edges;
every fist had at least 3.7% (median 9%). A hand needs BOTH tests: difference at or
above the threshold AND at least EDGE_MIN_PERCENT new edges.
"""
import cv2
import numpy as np

DEFAULT_THRESHOLD = 6.0
BLUR = 5
EDGE_MIN_PERCENT = 1.0   # empty squares: 0.00%; fists: >= 3.7%
EDGE_STRENGTH = 40       # Sobel magnitude of the blurred grayscale crop
EDGE_NEAR_PX = 7         # snapshot edges count within this window (small camera shifts)


def _gray(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(gray, (BLUR, BLUR), 0)


def difference(background_gray, crop):
    """Brightness-corrected mean absolute difference between a crop and the background."""
    change = _gray(crop) - background_gray
    return float(np.abs(change - np.median(change)).mean())


def _edges(gray):
    return cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)) > EDGE_STRENGTH


def new_edges(background_edges_near, crop):
    """Percent of pixels with a strong edge that the snapshot didn't have nearby."""
    return float((_edges(_gray(crop)) & ~background_edges_near).mean() * 100)


class EmptySquareCheck:
    def __init__(self, threshold=DEFAULT_THRESHOLD, edge_min=EDGE_MIN_PERCENT):
        self.threshold = threshold
        self.edge_min = edge_min
        self.background = None
        self.background_edges = None
        self.last_new_edges = None  # of the latest check, for display

    def set_background(self, crop):
        """Snapshot of the empty square (a BGR crop from crop_and_resize)."""
        self.background = _gray(crop)
        kernel = np.ones((EDGE_NEAR_PX, EDGE_NEAR_PX), np.uint8)
        self.background_edges = cv2.dilate(_edges(self.background).astype(np.uint8), kernel).astype(bool)

    def check(self, crop):
        """Returns (difference, is_empty). Before set_background: (None, False).

        Empty unless the crop differs from the snapshot (difference >= threshold) AND has
        new edges (>= edge_min %): drifted light alone never makes a hand.
        """
        if self.background is None:
            self.last_new_edges = None
            return None, False
        diff = difference(self.background, crop)
        self.last_new_edges = new_edges(self.background_edges, crop)
        return diff, diff < self.threshold or self.last_new_edges < self.edge_min
