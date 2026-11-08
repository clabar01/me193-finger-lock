"""Time the lock's per-frame work on saved crops, to compare the Mac with AirFour.

    python lock/benchmark.py --images data/bench --model models/finger_cnn.onnx
    (on the board, inside the app container: see board/benchmark_board.sh)

For each onnxruntime setting it runs, per image, exactly what the lock does each
frame: preprocess (common/preprocess.py), the no-hand check and the model. It
reports wall time per frame (mean, median, 95th percentile) and CPU time per
frame; CPU time per frame x 30 fps / 1000 = cores used at the camera's 30 fps.
It also prints the logits of the first image, to check both machines agree.
"""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "board" / "python", HERE):
    sys.path.insert(0, str(p))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from common.no_hand import EmptySquareCheck  # noqa: E402
from common.preprocess import preprocess  # noqa: E402


def make_session(model, threads, spin):
    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    options.add_session_config_entry("session.intra_op.allow_spinning", "1" if spin else "0")
    return ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])


def run(model, crops, threads, spin, repeats):
    session = make_session(model, threads, spin)
    name = session.get_inputs()[0].name
    mode = session.get_modelmeta().custom_metadata_map["preprocess_mode"]
    check = EmptySquareCheck()
    check.set_background(crops[0])
    for crop in crops[:10]:  # warm-up
        session.run(None, {name: preprocess(crop, mode=mode)[None]})
    wall, cpu = [], []
    for _ in range(repeats):
        for crop in crops:
            t, c = time.perf_counter(), time.process_time()
            session.run(None, {name: preprocess(crop, mode=mode)[None]})
            check.check(crop)
            wall.append(1000 * (time.perf_counter() - t))
            cpu.append(1000 * (time.process_time() - c))
    first = session.run(None, {name: preprocess(crops[0], mode=mode)[None]})[0][0]
    return {"threads": threads or "default", "spinning": spin,
            "ms_mean": round(statistics.mean(wall), 2), "ms_median": round(statistics.median(wall), 2),
            "ms_p95": round(float(np.percentile(wall, 95)), 2), "cpu_ms_per_frame": round(statistics.mean(cpu), 2),
            "cores_at_30fps": round(statistics.mean(cpu) * 30 / 1000, 2), "first_logits": np.round(first, 4).tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--settings", default="0:1,1:0,2:0,4:0", help="threads:spin pairs; 0 threads = default")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    crops = [cv2.imread(str(p)) for p in sorted(args.images.glob("*.png"))]
    if not crops:
        raise SystemExit(f"no PNGs in {args.images}")
    info = {"machine": platform.machine(), "system": platform.system(), "python": platform.python_version(),
            "onnxruntime": ort.__version__, "opencv": cv2.__version__, "images": len(crops)}
    print(json.dumps(info))
    for pair in args.settings.split(","):
        threads, spin = (int(x) for x in pair.split(":"))
        print(json.dumps(run(args.model, crops, threads, bool(spin), args.repeats)))


if __name__ == "__main__":
    main()
