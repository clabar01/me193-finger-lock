"""The one preprocessing path, shared by capture, training and the live lock.

    preprocess(frame, roi)   raw camera frame -> model input (live lock)
    preprocess(saved_crop)   saved PNG        -> the same model input (training)

Both give identical numbers for the same frame, because capture saves the output
of crop_and_resize() losslessly and preprocess() runs that step first. Only
OpenCV and NumPy are needed, so this also runs on the Uno Q without PyTorch:

    x = torch.from_numpy(preprocess(frame, roi)).unsqueeze(0)

Images are in OpenCV's BGR order on the way in, as cv2.VideoCapture and
cv2.imread give them.
"""
import json
from pathlib import Path

import cv2
import numpy as np

IMAGE_SIZE = 128
MEAN = 0.5
STD = 0.5
ROI_PATH = Path(__file__).resolve().parents[1] / "config" / "roi.json"


def load_roi(path=ROI_PATH):
    """Read the ROI saved by capture/calibrate_roi.py: fractions of the frame."""
    with open(path) as f:
        roi = json.load(f)
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    if not (w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= 1 and y + h <= 1):
        raise ValueError(f"{path}: ROI is outside the frame: {roi}")
    return roi


def roi_to_pixels(roi, frame_width, frame_height):
    """Convert a fractional ROI to integer pixels (x0, y0, x1, y1)."""
    x0 = round(roi["x"] * frame_width)
    y0 = round(roi["y"] * frame_height)
    x1 = round((roi["x"] + roi["w"]) * frame_width)
    y1 = round((roi["y"] + roi["h"]) * frame_height)
    return x0, y0, x1, y1


def crop_and_resize(frame, roi=None):
    """Crop to the ROI and resize to IMAGE_SIZE: a BGR uint8 image, what capture saves.

    roi=None uses the whole image, which leaves a saved crop unchanged.
    """
    if roi is not None:
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = roi_to_pixels(roi, width, height)
        frame = frame[y0:y1, x0:x1]
    if frame.shape[:2] == (IMAGE_SIZE, IMAGE_SIZE):
        return frame.copy()
    return cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def preprocess(frame, roi=None):
    """Turn a BGR frame into model input: float32 RGB, shape (3, 128, 128), range -1 to 1.

    Pass roi for a raw camera frame; leave it out for a saved crop.
    """
    rgb = cv2.cvtColor(crop_and_resize(frame, roi), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)
