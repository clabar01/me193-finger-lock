"""Mark the red square at a fixed 640x480 and save it to config/roi.json.

The green box is the ROI to save. It starts on the auto-detected red square, or
the previously saved ROI, and dragging replaces it with a square box. Yellow
shows what auto-detect currently sees.

Keys: S = save and quit, A = use the auto-detected box, Q = quit without saving.

Rerun this whenever the camera or the red square moves.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.camera import CameraError, HEIGHT, WIDTH, open_configured_camera  # noqa: E402
from common.draw import label  # noqa: E402
from common.preprocess import IMAGE_SIZE, ROI_PATH, load_roi, roi_to_pixels  # noqa: E402
from common.red_square import box_to_roi, find_red_square  # noqa: E402

import cv2  # noqa: E402

WINDOW = "calibrate_roi"
def drag_to_square(start, end, width, height):
    """The square with one corner at start, reaching toward end, kept inside the frame."""
    x_sign = 1 if end[0] >= start[0] else -1
    y_sign = 1 if end[1] >= start[1] else -1
    room_x = width - start[0] if x_sign > 0 else start[0]
    room_y = height - start[1] if y_sign > 0 else start[1]
    side = min(max(abs(end[0] - start[0]), abs(end[1] - start[1])), room_x, room_y)
    x_other, y_other = start[0] + x_sign * side, start[1] + y_sign * side
    return min(start[0], x_other), min(start[1], y_other), max(start[0], x_other), max(start[1], y_other)


def main():
    try:
        cap, config = open_configured_camera()
    except CameraError as e:
        print(e)
        return 1
    print(f"Camera: {config['name']} at {WIDTH}x{HEIGHT}")

    state = {"box": None, "start": None}
    if ROI_PATH.exists():
        state["box"] = roi_to_pixels(load_roi(), WIDTH, HEIGHT)

    def on_mouse(event, x, y, flags, param):
        x, y = min(max(x, 0), WIDTH - 1), min(max(y, 0), HEIGHT - 1)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["start"]:
            state["box"] = drag_to_square(state["start"], (x, y), WIDTH, HEIGHT)
        elif event == cv2.EVENT_LBUTTONUP:
            state["start"] = None

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("The camera stopped delivering frames: nothing saved.")
                return 1

            auto_box = find_red_square(frame)
            if state["box"] is None and auto_box:
                state["box"] = auto_box

            view = frame.copy()
            if auto_box:
                cv2.rectangle(view, auto_box[:2], auto_box[2:], (0, 255, 255), 1)
            if state["box"]:
                cv2.rectangle(view, state["box"][:2], state["box"][2:], (0, 255, 0), 2)
            label(view, "Drag = box   A = auto   S = save   Q = quit", (10, 28))
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("a"), ord("A")) and auto_box:
                state["box"] = auto_box
            elif key in (ord("q"), ord("Q")):
                print("Quit: nothing saved.")
                return 1
            elif key in (ord("s"), ord("S")) and state["box"]:
                box = state["box"]
                if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                    continue
                ROI_PATH.parent.mkdir(exist_ok=True)
                roi = box_to_roi(box, WIDTH, HEIGHT)
                ROI_PATH.write_text(json.dumps(roi, indent=2) + "\n")
                side = box[2] - box[0]
                print(f"Saved {ROI_PATH}: {side}x{side} pixels at ({box[0]}, {box[1]})")
                if side < IMAGE_SIZE:
                    print(f"Warning: the box is under {IMAGE_SIZE} pixels, so crops will be enlarged and soft.")
                return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
