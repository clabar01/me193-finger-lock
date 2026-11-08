"""Tests for the training data split, augmentation and model shape."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from training.data import GAP, Sample, augment, load_samples, split_samples  # noqa: E402
from training.model import FingerCNN  # noqa: E402


def fake_samples(persons=("ceci",), sessions=("s1",), per_label=50):
    samples = []
    for person in persons:
        for k, session in enumerate(sessions):
            for label in range(6):
                for i in range(per_label):
                    t = 1000 * k + 100 * label + 0.2 * i
                    samples.append(Sample(Path(f"{person}/{label}/{session}_{i}.png"), label, person, "R", session, t))
    return samples


class SplitTests(unittest.TestCase):
    def check_disjoint(self, train, val, test):
        paths = [{s.path for s in part} for part in (train, val, test)]
        self.assertFalse(paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])

    def test_one_session_holds_out_the_end_with_a_gap(self):
        samples = fake_samples()
        train, val, test, how = split_samples(samples)
        self.check_disjoint(train, val, test)
        self.assertIn("last 20%", how)
        for label in range(6):
            tr = sorted(s.time for s in train if s.label == label)
            va = sorted(s.time for s in val if s.label == label)
            te = sorted(s.time for s in test if s.label == label)
            self.assertEqual(len(te), 10)
            # At least GAP images (0.2 s apart) between train and val, and between val and test.
            self.assertGreaterEqual(va[0] - tr[-1], 0.2 * (GAP + 1) - 1e-6)
            self.assertGreater(te[0] - va[-1], 0.2 * GAP)
        self.assertEqual(len(train) + len(val) + len(test) + 6 * 2 * GAP, len(samples))

    def test_several_sessions_hold_out_whole_sessions(self):
        train, val, test, how = split_samples(fake_samples(sessions=("a", "b", "c", "d", "e")))
        self.check_disjoint(train, val, test)
        test_sessions = {s.session for s in test}
        self.assertEqual(len(test_sessions), 1)
        self.assertFalse(test_sessions & {s.session for s in train + val})

    def test_several_people_hold_out_one_person(self):
        train, val, test, how = split_samples(fake_samples(persons=("alex", "ceci")), test_person="alex")
        self.assertEqual({s.person for s in test}, {"alex"})
        self.assertEqual({s.person for s in train + val}, {"ceci"})
        with self.assertRaises(ValueError):
            split_samples(fake_samples(persons=("alex", "ceci")), test_person="sam")

    def test_load_samples_filters_hand_and_exclusions(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "raw"
            image = np.zeros((128, 128, 3), np.uint8)
            for name in ("p_R_s1_20261106-174446-420", "p_R_s1_20261106-174446-620", "p_L_s1_20261106-174447-000"):
                (raw / "p" / "3").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(raw / "p" / "3" / f"{name}.png"), image)
            exclude = Path(d) / "exclude.txt"
            exclude.write_text("# comment\np/3/p_R_s1_20261106-174446-420.png  # bad frame\n")
            samples, excluded = load_samples(raw, "R", exclude)
            self.assertEqual(excluded, 1)
            self.assertEqual([s.path.name for s in samples], ["p_R_s1_20261106-174446-620.png"])
            self.assertEqual(samples[0].label, 3)


class AugmentAndModelTests(unittest.TestCase):
    def test_augment_keeps_shape_and_changes_image(self):
        rng = np.random.default_rng(0)
        crop = np.random.default_rng(1).integers(0, 255, (128, 128, 3), dtype=np.uint8)
        out = augment(crop, rng)
        self.assertEqual(out.shape, crop.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertFalse(np.array_equal(out, crop))

    def test_strong_augment_changes_colors_more(self):
        crop = np.random.default_rng(1).integers(60, 200, (128, 128, 3), dtype=np.uint8)
        def spread(strength):
            rng = np.random.default_rng(0)
            return np.std([augment(crop, rng, strength).mean(axis=(0, 1)) for _ in range(200)], axis=0).mean()
        self.assertGreater(spread("strong"), 1.5 * spread("mild"))

    def test_final_split_keeps_every_session(self):
        samples = fake_samples(sessions=("s1", "s2"))
        train, val, test, how = split_samples(samples, holdout=False)
        self.assertEqual(test, [])
        self.assertEqual({s.session for s in train}, {"s1", "s2"})
        self.assertEqual(len(train) + len(val) + 2 * 6 * GAP, len(samples))

    def test_model_output_and_size(self):
        model = FingerCNN()
        logits = model(torch.zeros(2, 3, 128, 128))
        self.assertEqual(tuple(logits.shape), (2, 6))
        self.assertLess(sum(p.numel() for p in model.parameters()), 200_000)
        stages, logits = model.stage_outputs(torch.zeros(1, 3, 128, 128))
        self.assertEqual([tuple(s.shape[1:]) for s in stages], [(16, 64, 64), (32, 32, 32), (64, 16, 16)])
        self.assertEqual(tuple(FingerCNN(in_channels=1)(torch.zeros(2, 1, 128, 128)).shape), (2, 6))


if __name__ == "__main__":
    unittest.main()
