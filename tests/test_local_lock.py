"""Tests for the board's local-mode helpers that don't need the board: the live accuracy test."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "board" / "python")]

# local_lock imports the board runtime (arduino.app_utils, the WebUI brick); stand in for it here.
fake_arduino = types.ModuleType("arduino")
fake_utils = types.ModuleType("arduino.app_utils")
fake_utils.App, fake_utils.Bridge, fake_utils.Logger = mock.Mock(), mock.Mock(), mock.Mock()
fake_bricks = types.ModuleType("arduino.app_bricks")
fake_web = types.ModuleType("arduino.app_bricks.web_ui")
fake_web.WebUI = mock.Mock()
sys.modules.update({"arduino": fake_arduino, "arduino.app_utils": fake_utils,
                    "arduino.app_bricks": fake_bricks, "arduino.app_bricks.web_ui": fake_web})

import local_lock  # noqa: E402

N = 5


class LiveEvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "live_eval.jsonl"
        self.eval = local_lock.LiveEval(N, 0.8, self.path)
        self.now = 1000.0

    def feed(self, value, frames, confidence=0.95, no_hand=False):
        for _ in range(frames):
            self.now += 0.033
            self.eval.feed(self.now, value, confidence, no_hand)

    def test_records_the_first_stable_reading(self):
        with mock.patch.object(local_lock.time, "monotonic", return_value=self.now):
            self.eval.start(3)
        self.feed(None, 10, no_hand=True)  # hand not in yet: ignored
        self.feed(2, N - 1)                 # passing through 2 on the way: not stable
        self.feed(3, N)
        summary = self.eval.summary()
        self.assertEqual((summary["n"], summary["correct"]), (1, 1))
        self.assertIsNone(summary["expected"])
        self.feed(4, 3 * N)                 # after the trial, nothing more is recorded
        self.assertEqual(self.eval.summary()["n"], 1)

    def test_wrong_and_missed_trials(self):
        with mock.patch.object(local_lock.time, "monotonic", return_value=self.now):
            self.eval.start(4)
        self.feed(3, N)                     # stable but wrong
        with mock.patch.object(local_lock.time, "monotonic", return_value=self.now):
            self.eval.start(5)
        self.feed(5, 3 * N, confidence=0.5) # never confident enough
        self.now += local_lock.EVAL_TIMEOUT_S
        self.feed(5, 1, confidence=0.5)     # times out: a miss
        s = self.eval.summary()
        self.assertEqual((s["n"], s["correct"]), (2, 0))
        self.assertEqual(s["per_label"]["4"], {"n": 1, "correct": 0})
        self.assertEqual([t["predicted"] for t in s["last"]], [None, 3])
        lines = [json.loads(x) for x in self.path.read_text().splitlines()]
        self.assertEqual([(t["expected"], t["predicted"]) for t in lines], [(4, 3), (5, None)])

    def test_results_survive_a_restart_and_reset_clears_them(self):
        with mock.patch.object(local_lock.time, "monotonic", return_value=self.now):
            self.eval.start(1)
        self.feed(1, N)
        again = local_lock.LiveEval(N, 0.8, self.path)
        self.assertEqual(again.summary()["n"], 1)
        again.reset()
        self.assertEqual(again.summary()["n"], 0)
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
