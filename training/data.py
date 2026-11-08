"""Find the captured images, split them without leaking, and augment training crops.

Images come from data/raw/<person>/<label>/<person>_<hand>_<session>_<timestamp>.png.
Every image goes through common.preprocess.preprocess, the same code as the live lock.

Hold-out rules, in order:
  1. More than one person: hold out one whole person for testing.
  2. One person, several sessions: hold out a random 20% of sessions.
  3. One person, one session: hold out the last 20% (in time) of each label's
     take. Images are saved at 5 per second, so neighbors are near-copies;
     GAP images at each boundary are dropped so no test image is a near-copy
     of a training image.
Validation is always the last 15% (in time) of each training group, with the
same gap.
"""
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXCLUDE = ROOT / "data" / "exclude.txt"
LABELS = "012345"
TEST_FRACTION = 0.2
VAL_FRACTION = 0.15
GAP = 5  # about 1 s at 5 images per second


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    person: str
    hand: str
    session: str
    time: float


def read_exclusions(path=EXCLUDE):
    if not Path(path).exists():
        return set()
    lines = (line.split("#")[0].strip() for line in Path(path).read_text().splitlines())
    return {line for line in lines if line}


def load_samples(raw=RAW, hand="R", exclude=EXCLUDE):
    """Every image of this hand, minus the exclusion list. Returns (samples, number excluded)."""
    skip = read_exclusions(exclude)
    samples, excluded = [], 0
    for path in sorted(Path(raw).glob("*/[0-5]/*.png")):
        if path.relative_to(raw).as_posix() in skip:
            excluded += 1
            continue
        person, file_hand, session, stamp = path.stem.split("_")
        if file_hand != hand:
            continue
        time = datetime.strptime(stamp + "000", "%Y%m%d-%H%M%S-%f").timestamp()
        samples.append(Sample(path, int(path.parent.name), person, file_hand, session, time))
    return samples, excluded


def _tail_split(samples, fraction):
    """Per (person, session, label) group in time order: (head, tail, dropped).

    The tail is the last `fraction`; GAP images before it are dropped.
    """
    groups = {}
    for s in samples:
        groups.setdefault((s.person, s.session, s.label), []).append(s)
    head, tail, dropped = [], [], []
    for group in groups.values():
        group.sort(key=lambda s: s.time)
        n_tail = math.ceil(fraction * len(group))
        cut = len(group) - n_tail
        head += group[:max(cut - GAP, 0)]
        dropped += group[max(cut - GAP, 0):cut]
        tail += group[cut:]
    return head, tail, dropped


def split_samples(samples, test_person=None, seed=0, holdout=True):
    """Returns (train, val, test, description). holdout=False keeps no test set (final model)."""
    persons = sorted({s.person for s in samples})
    dropped = []
    if not holdout:
        rest, test = samples, []
        how = "no test set: every session is used for training (final model)"
    elif len(persons) > 1:
        test_person = test_person or persons[-1]
        if test_person not in persons:
            raise ValueError(f"no images from {test_person}; people found: {persons}")
        rest = [s for s in samples if s.person != test_person]
        test = [s for s in samples if s.person == test_person]
        how = f"held-out person: {test_person}"
    else:
        sessions = sorted({s.session for s in samples})
        if len(sessions) > 1:
            n_test = max(1, round(TEST_FRACTION * len(sessions)))
            test_sessions = sorted(random.Random(seed).sample(sessions, n_test))
            rest = [s for s in samples if s.session not in test_sessions]
            test = [s for s in samples if s.session in test_sessions]
            how = f"held-out sessions: {', '.join(test_sessions)}"
        else:
            rest, test, dropped = _tail_split(samples, TEST_FRACTION)
            how = (f"only one person and one session, so the test set is the last "
                   f"{TEST_FRACTION:.0%} of each label's take, with a {GAP}-image gap")
    train, val, val_dropped = _tail_split(rest, VAL_FRACTION)
    dropped += val_dropped
    how += f"; validation is the last {VAL_FRACTION:.0%} of each training take; {len(dropped)} gap images unused"
    return train, val, test, how


def count_by_label(samples):
    counts = [0] * len(LABELS)
    for s in samples:
        counts[s.label] += 1
    return counts


# Color augmentation ranges. "mild" is the original setting; "strong" covers
# lighting as different as s1 vs s2 (mean brightness 93-144, red/blue 1.1-1.3).
COLOR_JITTER = {
    "mild":   {"gain": 0.10, "brightness": 25, "contrast": 0.25, "saturation": 0.2, "gamma": 0.0},
    "strong": {"gain": 0.30, "brightness": 50, "contrast": 0.40, "saturation": 0.5, "gamma": 0.5},
}


def augment(crop, rng, strength="strong"):
    """A randomly changed copy of a BGR uint8 crop: same shape and type.

    Geometry: rotation up to 15 degrees, shift up to 8%, scale 0.9-1.1, and a
    horizontal flip half the time (so left hands are partly covered). Color,
    with ranges from COLOR_JITTER[strength]: a separate gain per channel (the
    webcam's unlocked white balance), brightness, contrast, saturation and gamma.
    """
    jitter = COLOR_JITTER[strength]
    height, width = crop.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rng.uniform(-15, 15), rng.uniform(0.9, 1.1))
    matrix[:, 2] += rng.uniform(-0.08, 0.08, 2) * (width, height)
    image = cv2.warpAffine(crop, matrix, (width, height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.5:
        image = image[:, ::-1]

    x = image.astype(np.float32)
    if jitter["gamma"]:
        gamma = np.exp(rng.uniform(-jitter["gamma"], jitter["gamma"]))   # about 0.6-1.65 when strong
        x = 255.0 * (x / 255.0) ** gamma
    x = x * rng.uniform(1 - jitter["gain"], 1 + jitter["gain"], 3)       # white balance
    mean = x.mean()
    x = (x - mean) * rng.uniform(1 - jitter["contrast"], 1 + jitter["contrast"]) + mean
    x = x + rng.uniform(-jitter["brightness"], jitter["brightness"])
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * rng.uniform(1 - jitter["saturation"], 1 + jitter["saturation"])
    return np.clip(x, 0, 255).astype(np.uint8)
