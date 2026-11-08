"""The one preprocessing path, shared by capture, training and the live lock.

    preprocess(frame, roi, mode)   raw camera frame -> model input (live lock)
    preprocess(saved_crop, mode=)  saved PNG        -> the same model input (training)

Both give identical numbers for the same frame, because capture saves the output
of crop_and_resize() losslessly and preprocess() runs that step first. Only
OpenCV and NumPy are needed, so this also runs on the Uno Q without PyTorch:

    x = torch.from_numpy(preprocess(frame, roi, mode)).unsqueeze(0)

Modes (the trained model records its mode; use that one, never a guess):
    "fixed"    RGB scaled to -1..1 with fixed constants. Brightness and color
               casts go straight into the model.
    "image"    RGB standardized by this image's own mean and std, which
               removes overall brightness and contrast.
    "channel"  each RGB channel standardized by its own mean and std, which
               also removes a color cast such as a white-balance shift.
    "gray"     grayscale (1 channel), standardized by its own mean and std.

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
EPS = 1.0  # in 0-255 pixel units; keeps a nearly flat image from blowing up
MODES = {"fixed": 3, "image": 3, "channel": 3, "gray": 1}  # mode -> input channels
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


def preprocess(frame, roi=None, mode="fixed"):
    """Turn a BGR frame into model input: float32, shape (channels, 128, 128).

    Pass roi for a raw camera frame; leave it out for a saved crop.
    """
    if mode not in MODES:
        raise ValueError(f"unknown preprocessing mode {mode!r}; use one of {list(MODES)}")
    crop = crop_and_resize(frame, roi)
    if mode == "gray":
        x = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)[None]
    else:
        x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)

    if mode == "fixed":
        x = (x / 255.0 - MEAN) / STD
    elif mode == "channel":
        x = (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + EPS)
    else:
        x = (x - x.mean()) / (x.std() + EPS)
    return np.ascontiguousarray(x, dtype=np.float32)
