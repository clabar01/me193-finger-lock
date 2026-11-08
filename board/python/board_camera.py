"""Open the webcam on AirFour the way the Mac did for the training captures.

On the Mac (common/camera.py) the camera ran at 640x480, auto exposure was given
3 s to settle and then locked, and white balance stayed automatic (it can't be
locked there). The model was trained on those images, so this does the same.

Linux (V4L2) can lock exposure, but this camera doesn't report the exposure its
auto mode chose: it always says 39, and a manual 39 is nearly black. So after
settling, this switches to manual and adjusts the exposure until the brightness
matches what auto exposure gave. Exposure is capped so the camera keeps ~30 fps.
"""
import time

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
SETTLE_S = 3.0
MAX_EXPOSURE = 330   # V4L2 units of 100 us: 33 ms, one frame at 30 fps
TOLERANCE = 3.0      # brightness units (0-255)


class CameraError(RuntimeError):
    pass


def _mean_brightness(cap, frames):
    values = []
    for _ in range(frames):
        ok, frame = cap.read()
        if ok:
            values.append(float(frame.mean()))
    if not values:
        raise CameraError("the camera stopped delivering frames")
    return float(np.mean(values[-3:]))


def open_camera(device="/dev/video0", log=print):
    """Returns (cap, settings). Raises CameraError if the camera can't give 640x480."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise CameraError(f"could not open {device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    ok, frame = cap.read()
    if not ok or frame.shape[:2] != (HEIGHT, WIDTH):
        cap.release()
        raise CameraError(f"{device} must give {WIDTH}x{HEIGHT} frames, got {frame.shape if ok else None}")

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # V4L2: 3 = auto (aperture priority)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    start, frames = time.monotonic(), 0
    while time.monotonic() - start < SETTLE_S:
        cap.read()
        frames += 1
    fps = frames / (time.monotonic() - start)
    target = _mean_brightness(cap, 10)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2: 1 = manual
    exposure, brightness = 200.0, None
    for _ in range(8):
        cap.set(cv2.CAP_PROP_EXPOSURE, round(exposure))
        brightness = _mean_brightness(cap, 6)
        if abs(brightness - target) <= TOLERANCE:
            break
        exposure = float(np.clip(exposure * target / max(brightness, 1.0), 1, MAX_EXPOSURE))
    settings = {
        "device": device, "width": WIDTH, "height": HEIGHT, "fourcc": "YUYV", "fps_measured": round(fps, 1),
        "auto_brightness": round(target, 1), "locked_brightness": round(brightness, 1),
        "exposure": f"locked at {round(exposure)} (x100 us)", "white_balance": "auto (as on the Mac)",
    }
    if abs(brightness - target) > TOLERANCE:
        log(f"exposure lock could not match auto brightness {target:.0f} (got {brightness:.0f})")
    return cap, settings
