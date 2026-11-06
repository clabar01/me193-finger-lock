# Data

The images are not on GitHub: `raw/` and `processed/` are ignored by git and raw
images are shared as a separate zip. This README and the empty folders are all
that is committed.

```
data/
├── raw/
│   └── <person>/
│       ├── 0/ 1/ 2/ 3/ 4/ 5/    one folder per finger count
│       │   └── <person>_<hand>_<session>_<timestamp>.png
│       └── sessions.jsonl       camera settings and ROI, one line per capture run
└── processed/                   generated files, such as contact sheets
```

## Labels

| Label | Meaning |
| --- | --- |
| 0 | Fist or empty square; separates the digits of a code |
| 1-5 | That many fingers; the digits of a code |

## Images

- 128x128 color PNG (lossless), cropped to the red square.
- Captured with the external USB webcam only, at 640x480, using `capture/capture.py`.
- In a filename, `hand` is `L` or `R`, and `session` is a free-form lighting note.
  Fields are separated by `_`, so names and notes use only letters, digits and hyphens.
  The timestamp is `YYYYMMDD-HHMMSS-mmm`.

## Using the images

Load every image through `common/preprocess.py`, the same code the live lock uses:

```python
from common.preprocess import preprocess

x = preprocess(cv2.imread(path))  # float32 RGB, shape (3, 128, 128), range -1 to 1
```

The label is the name of the image's parent folder.

## Zipping

```bash
cd data && zip -r ../finger_lock_raw.zip raw -x '*.gitkeep' '*.DS_Store'
```
