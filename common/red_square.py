"""Find the red square in a camera frame, for setting the ROI (Mac: capture/calibrate_roi.py,
AirFour: the Calibrate button on the board's web page). OpenCV and NumPy only."""
import cv2
import numpy as np

MIN_AREA_FRACTION = 0.005  # ignore red specks
MIN_FILL = 0.6             # of the bounding box: a red square fills it, clothing usually doesn't


def find_red_square(frame):
    """Return the most square-like red region as a square box (x0, y0, x1, y1), or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 90, 60), (10, 255, 255)) | cv2.inRange(hsv, (170, 90, 60), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = frame.shape[:2]
    best, best_area = None, MIN_AREA_FRACTION * width * height
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > best_area and 0.7 <= w / h <= 1.4 and area / (w * h) >= MIN_FILL:
            best, best_area = (x, y, w, h), area
    if best is None:
        return None
    x, y, w, h = best
    return clamp_square(x + w // 2, y + h // 2, max(w, h), width, height)


def clamp_square(center_x, center_y, side, width, height):
    """A square box of this side at this center, moved or shrunk to fit in the frame."""
    side = min(side, width, height)
    x0 = min(max(center_x - side // 2, 0), width - side)
    y0 = min(max(center_y - side // 2, 0), height - side)
    return x0, y0, x0 + side, y0 + side




def box_to_roi(box, width, height):
    """A pixel box (x0, y0, x1, y1) as the fractional ROI saved in roi.json."""
    x0, y0, x1, y1 = box
    return {"x": x0 / width, "y": y0 / height, "w": (x1 - x0) / width, "h": (y1 - y0) / height,
            "frame_width": width, "frame_height": height}
