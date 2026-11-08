"""Live test of the trained model with the USB webcam.

    python capture/live_test.py
    python capture/live_test.py --model models/finger_cnn.onnx --unsure 0.6 --empty-threshold 6

Runs the exported ONNX model, the file the board will use, on each frame
through common.preprocess.preprocess, in the preprocessing mode stored in the
ONNX file. Shows the ROI, the predicted finger count, its confidence and all
six probabilities.

It also runs the "no hand" check (common/no_hand.py): B takes a snapshot of
the empty square, standing in for the lock's button. After that, whenever the
square differs from the snapshot by less than the threshold, the answer is 0
whatever the model says. The difference is shown so the threshold can be tuned.

Keys: B = snapshot the empty square, + / - = raise / lower the threshold by 0.5, Q = quit.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.camera import CameraError, lock_and_report, open_configured_camera, restore_auto  # noqa: E402
from common.draw import label  # noqa: E402
from common.no_hand import DEFAULT_THRESHOLD, EmptySquareCheck  # noqa: E402
from common.preprocess import ROI_PATH, crop_and_resize, load_roi, preprocess, roi_to_pixels  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

WINDOW = "live_test"
PANEL_WIDTH = 320
THRESHOLD_STEP = 0.5
GREEN, AMBER, CYAN = (0, 220, 0), (0, 200, 255), (255, 200, 0)


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def draw(frame, box, probs, unsure, ms, diff, empty, threshold, note=""):
    """The feed with the ROI, beside a panel with the answer, the model's bars and the no-hand check."""
    view = frame.copy()
    best = int(probs.argmax())
    sure = probs[best] >= unsure
    if empty:
        headline, sub, color = "0 fingers", "no hand: snapshot match", CYAN
    elif sure:
        headline, sub, color = f"{best} fingers", f"model {probs[best]:.0%} confident", GREEN
    else:
        headline, sub, color = "unsure", f"model {probs[best]:.0%} confident", AMBER
    cv2.rectangle(view, box[:2], box[2:], color, 2)

    panel = np.full((view.shape[0], PANEL_WIDTH, 3), 40, np.uint8)
    label(panel, headline, (16, 56), 1.3, color, 3)
    label(panel, sub, (16, 94), 0.6)
    label(panel, "model:", (16, 128), 0.5)
    for i, p in enumerate(probs):
        y = 140 + i * 40
        label(panel, str(i), (16, y + 20), 0.7)
        cv2.rectangle(panel, (50, y), (50 + 200, y + 26), (80, 80, 80), -1)
        cv2.rectangle(panel, (50, y), (50 + int(200 * p), y + 26), (200, 200, 200) if i == best else (130, 130, 130), -1)
        label(panel, f"{p:.0%}", (260, y + 20), 0.55)

    if diff is None:
        label(panel, "no snapshot: empty the square, press B", (16, 400), 0.45, AMBER)
    else:
        label(panel, f"difference {diff:5.1f}   threshold {threshold:.1f}", (16, 400), 0.5,
              CYAN if empty else (255, 255, 255))
    if note:
        label(panel, note, (16, 428), 0.45, AMBER)
    label(panel, f"model {ms:.0f} ms   B snapshot  +/- threshold  Q quit", (16, view.shape[0] - 14), 0.42)
    return np.hstack((view, panel))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Live webcam test of the finger-count model.")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "finger_cnn.onnx")
    parser.add_argument("--unsure", type=float, default=0.6, help="below this confidence, show 'unsure'")
    parser.add_argument("--empty-threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"no-hand difference threshold (default {DEFAULT_THRESHOLD})")
    args = parser.parse_args(argv)

    import onnxruntime as ort

    if not args.model.exists():
        print(f"{args.model} not found: run training/train.py first")
        return 1
    if not ROI_PATH.exists():
        print(f"{ROI_PATH} not found: run capture/calibrate_roi.py first")
        return 1
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    mode = session.get_modelmeta().custom_metadata_map.get("preprocess_mode")
    if mode is None:
        print(f"{args.model} has no preprocess_mode in its metadata: re-export it with training/train.py")
        return 1
    print(f"Model: {args.model.name}, preprocessing mode {mode!r}")
    roi = load_roi()
    check = EmptySquareCheck(args.empty_threshold)

    try:
        cap, config = open_configured_camera()
    except CameraError as e:
        print(e)
        return 1
    note = ""
    try:
        settings = lock_and_report(cap, config)
        print(f"Camera: {settings['camera']}, exposure {settings['exposure']}, "
              f"white balance {settings['white_balance']}")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("The camera stopped delivering frames.")
                return 1
            crop = crop_and_resize(frame, roi)
            start = time.perf_counter()
            probs = softmax(session.run(None, {input_name: preprocess(crop, mode=mode)[None]})[0][0])
            ms = 1000 * (time.perf_counter() - start)
            diff, empty = check.check(crop)
            box = roi_to_pixels(roi, frame.shape[1], frame.shape[0])
            cv2.imshow(WINDOW, draw(frame, box, probs, args.unsure, ms, diff, empty, check.threshold, note))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print(f"Threshold at exit: {check.threshold:.1f}")
                return 0
            if key in (ord("b"), ord("B")):
                check.set_background(crop)
                # A hand in the square at snapshot time would make the hand the "empty" picture.
                note = "" if probs.argmax() == 0 else f"model saw {probs.argmax()} fingers at snapshot: retake?"
                print(f"Snapshot taken{': ' + note if note else ''}")
            elif key in (ord("+"), ord("=")):
                check.threshold += THRESHOLD_STEP
            elif key in (ord("-"), ord("_")):
                check.threshold = max(0.0, check.threshold - THRESHOLD_STEP)
    finally:
        restore_auto(cap, config.get("opened_index"))
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
