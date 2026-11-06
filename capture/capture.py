"""Capture labeled 128x128 crops of the red square from the USB webcam.

    python capture/capture.py --person ceci --hand R --session desklamp

Anything left out is asked for in the terminal.

Keys:
    0-5   save one image; hold the key to save about 5 per second
    B     burst mode on/off. When on, a digit starts saving that label at 5 per
          second, without holding it. Space stops; it also stops at the target.
    U     delete the last saved image (repeat to go further back)
    Q     quit

Files go to data/raw/<person>/<label>/<person>_<hand>_<session>_<timestamp>.png.
Camera settings for each run are added to data/raw/<person>/sessions.jsonl.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.camera import DARK_MEAN, CameraError, lock_and_report, open_configured_camera, restore_auto  # noqa: E402
from common.draw import label as text  # noqa: E402
from common.preprocess import IMAGE_SIZE, ROI_PATH, crop_and_resize, load_roi, roi_to_pixels  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

LABELS = "012345"
SAVE_INTERVAL_S = 0.2   # 5 images per second
UNDO_INTERVAL_S = 0.3   # so holding U doesn't wipe out a burst by accident
DEFAULT_TARGET = 200
WINDOW = "capture"
PREVIEW_SIZE = 480
PANEL_HEIGHT = 130


def clean_name(text):
    """Keep filenames parseable: letters, digits and hyphens only, since '_' separates fields."""
    return re.sub(r"[^A-Za-z0-9-]+", "-", text.strip()).strip("-")


class CropStore:
    """Saved crops for one person: counts per label, plus undo for this run."""

    def __init__(self, data_root, person, hand, session):
        self.folder = Path(data_root) / person
        self.prefix = f"{person}_{hand}_{session}"
        self.saved = []
        self.counts = {}
        for label in LABELS:
            (self.folder / label).mkdir(parents=True, exist_ok=True)
            self.counts[label] = len(list((self.folder / label).glob("*.png")))

    def save(self, label, crop):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        path = self.folder / label / f"{self.prefix}_{timestamp}.png"
        if not cv2.imwrite(str(path), crop):
            raise OSError(f"could not write {path}")
        self.saved.append((label, path))
        self.counts[label] += 1
        return path

    def undo(self):
        """Delete the newest image saved in this run. Returns its path, or None if none are left."""
        if not self.saved:
            return None
        label, path = self.saved.pop()
        path.unlink(missing_ok=True)
        self.counts[label] -= 1
        return path


def draw_screen(frame, crop, box, store, target, status, settings):
    """Live feed with the ROI, a big preview of exactly what gets saved, and the counts."""
    view = frame.copy()
    cv2.rectangle(view, box[:2], box[2:], (0, 255, 0), 2)
    height, width = view.shape[:2]
    big = cv2.resize(crop, (PREVIEW_SIZE, PREVIEW_SIZE), interpolation=cv2.INTER_NEAREST)
    panel = np.full((PANEL_HEIGHT, width + PREVIEW_SIZE, 3), 40, np.uint8)

    cell = (width + PREVIEW_SIZE) // len(LABELS)
    for i, label in enumerate(LABELS):
        count = store.counts[label]
        done = count >= target
        x = i * cell + 8
        fill = int((cell - 16) * min(count / target, 1))
        cv2.rectangle(panel, (x, 44), (x + cell - 16, 60), (90, 90, 90), -1)
        cv2.rectangle(panel, (x, 44), (x + fill, 60), (0, 200, 0) if done else (0, 170, 255), -1)
        text(panel, f"{label}: {count}/{target}", (x, 32), 0.7, (0, 255, 0) if done else (255, 255, 255), 2)

    text(panel, status, (8, 88), 0.65, (0, 255, 255), 1)
    text(panel, f"exposure {settings['exposure']}  |  white balance {settings['white_balance']}", (8, 118), 0.5)
    text(view, "0-5 save/hold  B burst  U undo  Q quit", (10, 28))
    text(big, "saved crop", (10, 28))
    return np.vstack((np.hstack((view, big[:height])), panel))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Capture labeled crops from the USB webcam.")
    parser.add_argument("--person")
    parser.add_argument("--hand", help="L or R")
    parser.add_argument("--session", help="session or lighting note, e.g. desklamp")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET, help="images per class (default 200)")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args(argv)
    if args.target < 1:
        parser.error("--target must be at least 1")

    person = clean_name(args.person or input("Person name: "))
    session = clean_name(args.session or input("Session/lighting note: "))
    hand = (args.hand or input("Hand (L/R): ")).strip().upper()
    if not person or not session:
        parser.error("person and session need at least one letter or digit")
    if hand not in ("L", "R"):
        parser.error("hand must be L or R")
    args.person, args.session, args.hand = person, session, hand
    return args


def main(argv=None):
    args = parse_args(argv)
    if not ROI_PATH.exists():
        print(f"{ROI_PATH} not found: run capture/calibrate_roi.py first")
        return 1
    roi = load_roi()

    try:
        cap, config = open_configured_camera()
    except CameraError as e:
        print(e)
        return 1

    try:
        print("Keep your hand out of view: letting the camera settle, then locking what it allows...")
        settings = lock_and_report(cap, config)
        print("Camera settings used:")
        for name, value in settings.items():
            print(f"  {name}: {value}")
        if settings["mean_brightness_at_lock"] < DARK_MEAN:
            print("WARNING: the image is very dark. Check the preview before saving anything.")

        store = CropStore(args.data_root, args.person, args.hand, args.session)
        with open(store.folder / "sessions.jsonl", "a") as f:
            f.write(json.dumps({"started": datetime.now().isoformat(timespec="seconds"), "person": args.person,
                                "hand": args.hand, "session": args.session, "roi": roi,
                                "camera": settings}) + "\n")

        burst_mode = False
        burst_label = None
        last_save = last_undo = 0.0
        status = "Ready"
        while True:
            ok, frame = cap.read()
            if not ok:
                print("The camera stopped delivering frames.")
                return 1
            height, width = frame.shape[:2]
            crop = crop_and_resize(frame, roi)
            box = roi_to_pixels(roi, width, height)
            now = time.monotonic()

            if burst_label is not None:
                if store.counts[burst_label] >= args.target:
                    status = f"Burst stopped: label {burst_label} reached {args.target}"
                    burst_label = None
                elif now - last_save >= SAVE_INTERVAL_S:
                    store.save(burst_label, crop)
                    last_save = now
                    status = f"BURST label {burst_label}: Space stops"

            cv2.imshow(WINDOW, draw_screen(frame, crop, box, store, args.target, status, settings))
            key = cv2.waitKey(1) & 0xFF
            char = chr(key).upper() if key != 255 else ""

            if char == "Q":
                break
            elif char == "U":
                if now - last_undo >= UNDO_INTERVAL_S:
                    burst_label = None
                    last_undo = now
                    removed = store.undo()
                    status = f"Deleted {removed.name}" if removed else "Nothing to undo in this run"
            elif char == "B":
                burst_mode = not burst_mode
                burst_label = None
                status = "Burst mode ON: press a digit to start" if burst_mode else "Burst mode off"
            elif char == " ":
                if burst_label is not None:
                    burst_label = None
                    status = "Burst stopped"
            elif char and char in LABELS:
                if burst_mode:
                    if burst_label is None:
                        burst_label = char
                elif now - last_save >= SAVE_INTERVAL_S:
                    path = store.save(char, crop)
                    last_save = now
                    status = f"Saved {path.name}"

        print("Images per class:", store.counts)
        print(f"Saved in this run: {len(store.saved)}")
        return 0
    finally:
        restore_auto(cap, config.get("opened_index"))
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
