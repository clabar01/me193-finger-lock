"""Guards: while unlocked, the ONLY way to lock is a fist held 3 s. No timer anywhere.

If any of these fail, a timer-based relock (or something that acts like one) came back:
  * the policy relocks by itself after any amount of time without a fist
  * a settings file or PolicyConfig has a relock/auto-lock timer
  * the sketch relocks while unlocked for any reason other than the crash/disconnect
    failsafe (no word from the app for APP_SILENT_MS), or that failsafe could fire while
    the app runs (its "alive" heartbeat is too slow)
  * an empty square whose light drifts after the snapshot reads as a fist and counts the
    lock down (the relocks seen on AirFour on 2026-11-09: no-hand diff 6.6-10.4, CNN "0")
"""
import dataclasses
import json
import re
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "board" / "python")]

from common.lock_policy import NO_HAND, UNLOCKED, LOCKED, LockPolicy, PolicyConfig, frame_reading  # noqa: E402
from common.lock_runtime import ALIVE_PERIOD_S, TIMER_WORDS, Heartbeat, check_settings  # noqa: E402
from common.no_hand import EmptySquareCheck  # noqa: E402

SKETCH = (ROOT / "board" / "sketch" / "sketch.ino").read_text()
CODE = (2, 4, 1)
N = 10


def unlocked_policy():
    p = LockPolicy(PolicyConfig(code=CODE, stable_frames=N))
    t = 0.0
    p.press(t)
    for i, d in enumerate(CODE):
        for v in ((d, 0) if i < len(CODE) - 1 else (d,)):
            for _ in range(N + 2):
                t += 1 / 30
                p.update(t, v, 0.95)
    assert p.state == UNLOCKED
    return p, t


# ---------- synthetic camera crops (128 x 128 BGR), like crop_and_resize gives ----------

def paper():
    """The empty square: off-white paper, a soft vignette, the red frame and the paper's edge."""
    yy, xx = np.mgrid[0:128, 0:128]
    base = 205 - 0.004 * ((xx - 64) ** 2 + (yy - 64) ** 2)
    im = np.dstack([base * 0.95, base, base * 1.02]).astype(np.float32)
    im[:, 118:] = (60, 40, 210)            # red frame on the right
    im[119:, :] = (60, 40, 210)            # and at the bottom
    cv2.line(im, (110, 0), (120, 127), (150, 150, 150), 2)  # the paper's slanted edge
    return np.clip(im, 0, 255).astype(np.uint8)


def drift(im, gain, tint, shadow):
    """Light, white-balance and a soft shadow change: what happens after the snapshot."""
    x = im.astype(np.float32) * gain * np.array(tint, np.float32)
    yy, xx = np.mgrid[0:128, 0:128]
    x *= (1 - shadow * np.clip((xx + yy) / 256 * 2 - 0.5, 0, 1))[..., None]
    return np.clip(x, 0, 255).astype(np.uint8)


def fist(im, dx=0):
    """A closed fist: a skin-coloured blob with an outline and knuckle creases."""
    im = im.copy()
    c = (60 + dx, 70)
    cv2.ellipse(im, c, (34, 26), 0, 0, 360, (120, 150, 200), -1)
    cv2.ellipse(im, c, (34, 26), 0, 0, 360, (70, 90, 130), 2)
    for k in range(4):
        cv2.line(im, (36 + dx + 14 * k, 52), (40 + dx + 14 * k, 76), (80, 100, 140), 2)
    return im


class PolicyHasNoTimer(unittest.TestCase):
    def test_policy_config_has_no_timer_field(self):
        fields = [f.name for f in dataclasses.fields(PolicyConfig)]
        self.assertEqual([f for f in fields if any(w in f.lower() for w in TIMER_WORDS)], [])

    def test_unlocked_survives_six_hours_of_anything_but_a_held_fist(self):
        p, t = unlocked_policy()
        events = []
        rng = np.random.default_rng(1)
        # 6 hours at 2 frames/s: no hand, open hands, unsure fists, short fists (< 3 s)
        for i in range(6 * 3600 * 2):
            t += 0.5
            r = rng.random()
            if r < 0.5:
                value, conf = NO_HAND, 0.0
            elif r < 0.8:
                value, conf = int(rng.integers(1, 6)), 0.95
            else:
                value, conf = 0, 0.5  # an unsure "fist"
            events += p.update(t, value, conf)
            if i % 600 == 0:
                for _ in range(30):     # a real fist for 1 s, then let go
                    t += 1 / 30
                    events += p.update(t, 0, 0.99)
            events += p.tick(t)
        self.assertEqual(p.state, UNLOCKED)
        self.assertIsNone(p.seconds_left(t))
        self.assertNotIn("fist_lock", [e.kind for e in events])
        self.assertEqual([e.kind for e in events if e.kind not in ("countdown", "countdown_cancel")], [])
        p.tick(t + 10 ** 7)              # months later
        self.assertEqual(p.state, UNLOCKED)


class SettingsHaveNoTimer(unittest.TestCase):
    def test_config_files_pass_and_timer_settings_are_rejected(self):
        for path in (ROOT / "config" / "lock.example.json", ROOT / "config" / "lock.json"):
            if path.exists():
                check_settings(json.loads(path.read_text()))
        good = json.loads((ROOT / "config" / "lock.example.json").read_text())
        for key in ("relock_s", "auto_lock_s", "unlock_timeout"):
            with self.assertRaisesRegex(ValueError, "timer"):
                check_settings({**good, key: 120})
        with self.assertRaisesRegex(ValueError, "unknown"):
            check_settings({**good, "something_old": 1})


class SketchHasOnlyTheFailsafe(unittest.TestCase):
    def update_mode_unlocked_block(self):
        body = SKETCH[SKETCH.index("void updateMode("):]
        return body[body.index("case M_LOCKED:"):body.index("case M_FLASH:")]

    def test_no_relock_timer_in_the_sketch(self):
        self.assertIsNone(re.search(r"relockAt|RELOCK_MS|DEFAULT_RELOCK|relock_s", SKETCH))
        unlock = SKETCH[SKETCH.index('!strcmp(c.name, "unlock")'):]
        unlock = unlock[:unlock.index("} else if")]
        self.assertNotRegex(unlock, r"millis\(\)|c\.a")  # unlock sets no time

    def test_only_relock_while_unlocked_is_the_app_silence_failsafe(self):
        block = self.update_mode_unlocked_block()
        self.assertEqual(block.count("setMode(M_LOCKED)"), 1, block)
        self.assertIn("mode == M_UNLOCKED && elapsed(lastWordMs, now) >= (long)APP_SILENT_MS", block)

    def test_failsafe_never_fires_while_the_app_runs(self):
        silent_ms = int(re.search(r"APP_SILENT_MS\s*=\s*(\d+)", SKETCH).group(1))
        self.assertGreaterEqual(silent_ms, 6 * ALIVE_PERIOD_S * 1000)  # 6 missed heartbeats needed
        # The app's frame loop (~30 fps, with hiccups of up to 2 s) for 3 hours, unlocked.
        beat, t, last_word, worst = Heartbeat(), 0.0, 0.0, 0.0
        rng = np.random.default_rng(2)
        while t < 3 * 3600:
            t += 1 / 30 + (2.0 if rng.random() < 0.001 else 0.0)
            if beat.due(t):
                worst = max(worst, t - last_word)
                last_word = t
        self.assertLess(worst * 1000, silent_ms)
        self.assertLess(worst, ALIVE_PERIOD_S + 2.1)
        self.assertIn('if (!strcmp(c.name, "alive")) return;', SKETCH)  # alive refreshes lastWordMs first
        self.assertLess(SKETCH.index("lastWordMs = millis();\n  if (!strcmp(c.name, \"alive\"))"),
                        SKETCH.index('if (!strcmp(c.name, "alive")) return;') + 1)


class DriftedEmptySquareIsNotAFist(unittest.TestCase):
    """The AirFour relocks of 2026-11-09, end to end: CNN says 0 for the empty square."""

    def test_drifted_empty_square_never_counts_down(self):
        check = EmptySquareCheck(6.0)
        check.set_background(paper())
        p, t = unlocked_policy()
        events, rng, old_rule_hand = [], np.random.default_rng(3), 0
        for _ in range(10):             # the hand taken away after the code: the square is clear
            t += 0.1
            events += p.update(t, *frame_reading([0.99] + [0.002] * 5, check.check(paper())[1]))
        light = (1.0, np.ones(3), 0.0)
        for i in range(10 * 60 * 10):   # 10 minutes at 10 fps of an empty square in drifting light
            t += 0.1
            if i % 50 == 0:              # the light changes every 5 s and then stays (like a shadow)
                light = (rng.uniform(0.8, 1.2), rng.uniform(0.88, 1.12, 3), rng.uniform(0, 0.4))
            crop = drift(paper(), *light)
            diff, empty = check.check(crop)
            old_rule_hand += diff >= 6.0
            value, conf = frame_reading([0.99, 0.002, 0.002, 0.002, 0.002, 0.002], empty)
            events += p.update(t, value, conf)
        self.assertGreater(old_rule_hand, 100)  # the old difference-only check would have seen a hand
        self.assertEqual(p.state, UNLOCKED)
        self.assertNotIn("countdown", [e.kind for e in events])

    def test_a_real_fist_still_locks(self):
        check = EmptySquareCheck(6.0)
        check.set_background(paper())
        p, t = unlocked_policy()
        for _ in range(30):              # hand away for 1 s (releases, past nothing yet)
            t += 0.1
            p.update(t, *frame_reading([0.99] + [0.002] * 5, check.check(paper())[1]))
        events = []
        for k in range(40):              # a fist, moving a little, in drifting light, 4 s
            t += 0.1
            crop = drift(fist(paper(), dx=k % 3), 1.1, (1.05, 1.0, 0.95), 0.2)
            diff, empty = check.check(crop)
            self.assertFalse(empty, (diff, check.last_new_edges))
            events += p.update(t, *frame_reading([0.99] + [0.002] * 5, empty))
        self.assertEqual(p.state, LOCKED)
        self.assertIn("fist_lock", [e.kind for e in events])


if __name__ == "__main__":
    unittest.main()
