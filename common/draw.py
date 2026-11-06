"""Readable on-screen text for the capture tools."""
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(image, text, position, scale=0.6, color=(255, 255, 255), thickness=1):
    """Draw text on a black background; position is the left end of the baseline."""
    if not text:
        return
    (width, height), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = position
    cv2.rectangle(image, (x - 3, y - height - 3), (x + width + 3, y + baseline + 1), (0, 0, 0), -1)
    cv2.putText(image, text, position, FONT, scale, color, thickness, cv2.LINE_AA)
