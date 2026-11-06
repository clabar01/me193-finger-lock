"""Preview each camera, confirm which is the USB webcam, and save it to config/camera.json.

Keys in the preview: Y = use this camera, N = next camera, Q = quit without saving.
On a Mac, Y only works on a USB camera, so the MacBook camera and Continuity
Camera can't be chosen.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.camera import CAMERA_PATH, HEIGHT, WIDTH, CameraError, list_mac_devices, open_index  # noqa: E402
from common.draw import label  # noqa: E402

import cv2  # noqa: E402

WINDOW = "list_cameras"
LINUX_MAX_INDEX = 10


def find_cameras():
    """Every camera to try. Only a Mac knows names and connection types."""
    if sys.platform == "darwin":
        return list_mac_devices()
    return [{"index": i, "name": f"camera {i}", "unique_id": None, "is_usb": True}
            for i in range(LINUX_MAX_INDEX)]


def preview(camera):
    """Show one camera until a key decides: returns "yes", "next" or "quit"."""
    try:
        cap = open_index(camera["index"])
    except CameraError as e:
        print(f"  index {camera['index']} ({camera['name']}): skipped: {e}")
        return "next"

    print(f"  index {camera['index']}: {camera['name']} ({'USB' if camera['is_usb'] else 'not USB'})")
    message = ""
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"  index {camera['index']}: stopped delivering frames")
                return "next"
            title = f"{camera['index']}: {camera['name']}"
            hint = "Y = use this   N = next   Q = quit" if camera["is_usb"] else "NOT USB: N = next   Q = quit"
            for y, text, color in ((28, title, (0, 255, 255)), (56, hint, (255, 255, 255)),
                                   (84, message, (0, 0, 255))):
                label(frame, text, (10, y), 0.7, color, 2)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("y"), ord("Y")):
                if camera["is_usb"]:
                    return "yes"
                message = "Not a USB camera: the assignment needs the webcam"
            elif key in (ord("n"), ord("N")):
                return "next"
            elif key in (ord("q"), ord("Q")):
                return "quit"
    finally:
        cap.release()


def main():
    cameras = find_cameras()
    if not cameras:
        print("No cameras found. Check the USB cable, then System Settings > Privacy & Security > Camera.")
        return 1

    print(f"Found {len(cameras)} camera(s). Previewing each in turn:")
    try:
        while True:
            for camera in cameras:
                choice = preview(camera)
                if choice == "quit":
                    print("Quit: nothing saved.")
                    return 1
                if choice == "yes":
                    CAMERA_PATH.parent.mkdir(exist_ok=True)
                    config = {**camera, "width": WIDTH, "height": HEIGHT}
                    CAMERA_PATH.write_text(json.dumps(config, indent=2) + "\n")
                    print(f"Saved {camera['name']} to {CAMERA_PATH}")
                    return 0
            print("Back to the first camera.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
