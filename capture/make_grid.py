"""Save a contact sheet of one person's newest crops: one row per label.

    python capture/make_grid.py --person test

Writes data/processed/grid_<person>.png, which stays out of git with the data.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.preprocess import IMAGE_SIZE  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

LABELS = "012345"
LABEL_WIDTH = 150
GAP = 4


def build_grid(person_folder, per_class):
    """Returns (grid image, {label: total count}, list of files that aren't 128x128 color)."""
    rows, counts, bad = [], {}, []
    for label in LABELS:
        paths = sorted((person_folder / label).glob("*.png"))
        counts[label] = len(paths)
        row = np.full((IMAGE_SIZE + 2 * GAP, LABEL_WIDTH + per_class * (IMAGE_SIZE + GAP), 3), 40, np.uint8)
        cv2.putText(row, f"{label}  ({len(paths)})", (10, IMAGE_SIZE // 2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        for i, path in enumerate(paths[-per_class:]):
            image = cv2.imread(str(path))
            if image is None or image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                bad.append(path)
                continue
            x = LABEL_WIDTH + i * (IMAGE_SIZE + GAP)
            row[GAP:GAP + IMAGE_SIZE, x:x + IMAGE_SIZE] = image
        rows.append(row)
    return np.vstack(rows), counts, bad


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--person", required=True)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    folder = args.data_root / args.person
    if not folder.is_dir():
        print(f"{folder} not found")
        return 1
    grid, counts, bad = build_grid(folder, args.per_class)
    out = args.out or ROOT / "data" / "processed" / f"grid_{args.person}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), grid):
        print(f"could not write {out}")
        return 1
    print(f"Images per class: {counts}")
    print(f"Wrote {out}")
    for path in bad:
        print(f"Bad image (not {IMAGE_SIZE}x{IMAGE_SIZE} color): {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
