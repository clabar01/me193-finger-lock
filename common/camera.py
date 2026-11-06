"""Open the external webcam the same way everywhere, at a fixed 640x480.

On macOS the camera is found by its saved unique ID, not its index, because the
indexes change when a camera (such as an iPhone) appears or disappears. OpenCV
sorts devices by unique ID, so list_mac_devices() uses the same order.

OpenCV can't change exposure or white balance on macOS, so those go through
AVFoundation. The Linux (V4L2) path is for the Uno Q and is untested so far.

A lock outlives the program on this webcam: reopened while still locked, it has
no auto exposure to settle with and the frames come out nearly black. So opening
a camera always returns it to auto, and lock_and_report() locks from there.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402

WIDTH = 640
HEIGHT = 480
SETTLE_S = 3.0  # auto exposure took about 1.8 s to recover from dark on this webcam
DARK_MEAN = 30  # below this mean brightness (0-255) at lock time, warn
CAMERA_PATH = Path(__file__).resolve().parents[1] / "config" / "camera.json"


class CameraError(RuntimeError):
    pass


def backend():
    return cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_V4L2


def _mac_avfoundation_devices():
    import AVFoundation as av

    devices = list(av.AVCaptureDevice.devicesWithMediaType_(av.AVMediaTypeVideo))
    devices += list(av.AVCaptureDevice.devicesWithMediaType_(av.AVMediaTypeMuxed))
    return sorted(devices, key=lambda d: str(d.uniqueID()))


def list_mac_devices():
    """Cameras in OpenCV's index order: index, name, unique_id, is_usb."""
    usb = int.from_bytes(b"usb ", "big")
    return [
        {"index": i, "name": str(d.localizedName()), "unique_id": str(d.uniqueID()),
         "is_usb": d.transportType() == usb}
        for i, d in enumerate(_mac_avfoundation_devices())
    ]


def _set_mac_modes(device, exposure_mode, wb_mode):
    """Apply whichever modes this device supports. Returns the modes it now reports."""
    can_expose = device.isExposureModeSupported_(exposure_mode)
    can_wb = device.isWhiteBalanceModeSupported_(wb_mode)
    if can_expose or can_wb:
        ok, error = device.lockForConfiguration_(None)
        if not ok:
            raise CameraError(f"could not configure the camera: {error}")
        if can_expose:
            device.setExposureMode_(exposure_mode)
        if can_wb:
            device.setWhiteBalanceMode_(wb_mode)
        device.unlockForConfiguration()
    return can_expose and device.exposureMode() == exposure_mode, can_wb and device.whiteBalanceMode() == wb_mode


def restore_auto(cap=None, index=None):
    """Put the camera back on auto exposure and auto white balance."""
    if sys.platform == "darwin":
        import AVFoundation as av

        devices = _mac_avfoundation_devices()
        if index >= len(devices):  # unplugged while running
            return
        _set_mac_modes(devices[index], av.AVCaptureExposureModeContinuousAutoExposure,
                       av.AVCaptureWhiteBalanceModeContinuousAutoWhiteBalance)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # V4L2: 3 = auto
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)


def open_index(index):
    """Open a camera index at 640x480 on auto exposure. Raises CameraError if it can't deliver that."""
    cap = cv2.VideoCapture(index, backend())
    if not cap.isOpened():
        cap.release()
        raise CameraError(f"camera {index} did not open. On a Mac, check System Settings > "
                          "Privacy & Security > Camera for the app running this script.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    restore_auto(cap, index)
    ok, frame = cap.read()
    if not ok or frame.shape[:2] != (HEIGHT, WIDTH):
        shape = frame.shape if ok else None
        cap.release()
        raise CameraError(f"camera {index} must give {WIDTH}x{HEIGHT} frames, got {shape}")
    return cap


def load_camera_config(path=CAMERA_PATH):
    if not Path(path).exists():
        raise CameraError(f"{path} not found: run capture/list_cameras.py first")
    with open(path) as f:
        return json.load(f)


def open_configured_camera(path=CAMERA_PATH):
    """Open the camera chosen with list_cameras.py. Returns (cap, config)."""
    config = load_camera_config(path)
    if sys.platform != "darwin":
        return open_index(config["index"]), config
    for device in list_mac_devices():
        if device["unique_id"] == config["unique_id"]:
            return open_index(device["index"]), {**config, "opened_index": device["index"]}
    raise CameraError(f"{config['name']} is not connected. Plug it in, or rerun capture/list_cameras.py.")


def _lock_mac(unique_id):
    import AVFoundation as av

    device = next(d for d in _mac_avfoundation_devices() if str(d.uniqueID()) == unique_id)
    locked_exposure, locked_wb = _set_mac_modes(device, av.AVCaptureExposureModeLocked,
                                                av.AVCaptureWhiteBalanceModeLocked)
    return {"exposure": "locked" if locked_exposure else "AUTO (cannot lock on this camera)",
            "white_balance": "locked" if locked_wb else "AUTO (cannot lock on this camera)"}


def _lock_v4l2(cap):
    settings = {}
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    # V4L2: 1 = manual exposure
    if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) and cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) == 1:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        settings["exposure"] = f"locked at {cap.get(cv2.CAP_PROP_EXPOSURE)}"
    else:
        settings["exposure"] = "AUTO (cannot lock on this camera)"
    temperature = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
    if cap.set(cv2.CAP_PROP_AUTO_WB, 0) and cap.get(cv2.CAP_PROP_AUTO_WB) == 0:
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, temperature)
        settings["white_balance"] = f"locked at {cap.get(cv2.CAP_PROP_WB_TEMPERATURE)}"
    else:
        settings["white_balance"] = "AUTO (cannot lock on this camera)"
    return settings


def lock_and_report(cap, config):
    """Let auto exposure settle, lock what the camera allows, and return the settings used.

    A "locked" result means the driver accepted the lock, not that the image was
    seen to stay constant.
    """
    start = time.monotonic()
    frame = None
    while time.monotonic() - start < SETTLE_S:
        ok, latest = cap.read()
        frame = latest if ok else frame
    if frame is None:
        raise CameraError("the camera delivered no frames while settling")
    locks = _lock_mac(config["unique_id"]) if sys.platform == "darwin" else _lock_v4l2(cap)
    return {
        "camera": config.get("name", f"index {config.get('index')}"),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps_reported": round(cap.get(cv2.CAP_PROP_FPS), 1),
        "backend": cap.getBackendName(),
        "platform": sys.platform,
        "mean_brightness_at_lock": round(float(frame.mean()), 1),
        **locks,
    }
