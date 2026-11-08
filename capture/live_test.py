"""Live test of the trained model with the USB webcam.

    python capture/live_test.py
    python capture/live_test.py --model models/finger_cnn.onnx --unsure 0.6

Runs the exported ONNX model, the file the board will use, on each frame
through common.preprocess.preprocess, in the preprocessing mode stored in the
ONNX file. Shows the ROI, the predicted finger count, its confidence and all
six probabilities. Q quits.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.camera import CameraError, lock_and_report, open_configured_camera, restore_auto  # noqa: E402
from common.draw import label  # noqa: E402
from common.preprocess import ROI_PATH, load_roi, preprocess, roi_to_pixels  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

WINDOW = "live_test"
PANEL_WIDTH = 320


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def draw(frame, box, probs, unsure, ms):
    """The feed with the ROI, beside a panel with the prediction and a bar per class."""
    view = frame.copy()
    best = int(probs.argmax())
    sure = probs[best] >= unsure
    color = (0, 220, 0) if sure else (0, 200, 255)
    cv2.rectangle(view, box[:2], box[2:], color, 2)

    panel = np.full((view.shape[0], PANEL_WIDTH, 3), 40, np.uint8)
    label(panel, "unsure" if not sure else f"{best} fingers", (16, 60), 1.3, color, 3)
    label(panel, f"{probs[best]:.0%} confident", (16, 100), 0.7)
    for i, p in enumerate(probs):
        y = 150 + i * 46
        label(panel, str(i), (16, y + 22), 0.8)
        cv2.rectangle(panel, (50, y), (50 + 200, y + 28), (80, 80, 80), -1)
        cv2.rectangle(panel, (50, y), (50 + int(200 * p), y + 28), color if i == best else (180, 180, 180), -1)
        label(panel, f"{p:.0%}", (260, y + 22), 0.6)
    label(panel, f"model {ms:.0f} ms/frame   Q = quit", (16, view.shape[0] - 16), 0.5)
    return np.hstack((view, panel))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Live webcam test of the finger-count model.")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "finger_cnn.onnx")
    parser.add_argument("--unsure", type=float, default=0.6, help="below this confidence, show 'unsure'")
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

    try:
        cap, config = open_configured_camera()
    except CameraError as e:
        print(e)
        return 1
    try:
        settings = lock_and_report(cap, config)
        print(f"Camera: {settings['camera']}, exposure {settings['exposure']}, "
              f"white balance {settings['white_balance']}")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("The camera stopped delivering frames.")
                return 1
            start = time.perf_counter()
            logits = session.run(None, {input_name: preprocess(frame, roi, mode)[None]})[0][0]
            ms = 1000 * (time.perf_counter() - start)
            box = roi_to_pixels(roi, frame.shape[1], frame.shape[0])
            cv2.imshow(WINDOW, draw(frame, box, softmax(logits), args.unsure, ms))
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q")):
                return 0
    finally:
        restore_auto(cap, config.get("opened_index"))
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
