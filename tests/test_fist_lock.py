"""Fist-to-lock while unlocked (common/lock_policy.py), its commands (common/lock_runtime.py) and sounds."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "board" / "python")]

import sounds  # noqa: E402
from common.lock_policy import LOCKED, NO_HAND, UNLOCKED, LockPolicy, PolicyConfig  # noqa: E402
from common.lock_runtime import handle_events  # noqa: E402

FPS = 30
N = 10
CODE = (2, 4, 1)


class Driver:
    """Unlocks the policy, then feeds frames at 30 fps; now = seconds since the unlock."""

    def __init__(self, **config):
        self.policy = LockPolicy(PolicyConfig(code=CODE, stable_frames=N, **config))
        self.t = 0.0
        self.policy.press(self.t)
        for i, d in enumerate(CODE):
            self.frames(d, N + 2)
            if i < len(CODE) - 1:
                self.frames(0, N + 2)  # fist between digits
        assert self.policy.state == UNLOCKED
        self.unlocked = self.policy.unlocked_at
        self.events = []

    @property
    def now(self):
        return self.t - self.unlocked

    def frames(self, value, count, confidence=0.95):
        out = []
        for _ in range(count):
            self.t += 1 / FPS
            out += self.policy.update(self.t, value, confidence)
        if hasattr(self, "events"):
            self.events += out
        return out

    def hold(self, value, seconds, confidence=0.95):
        return self.frames(value, round(seconds * FPS), confidence)

    def kinds(self):
        return [e.kind for e in self.events]

    def countdown(self):
        return [e.data["seconds"] for e in self.events if e.kind == "countdown"]


class FistLockTests(unittest.TestCase):
    def test_fist_held_3_s_locks_with_a_3_2_1_countdown(self):
        d = Driver()
        d.hold(3, 2.5)                  # past the grace period, hand open
        d.hold(0, 2.9)
        self.assertEqual(d.policy.state, UNLOCKED)
        self.assertEqual(d.countdown(), [3, 2, 1])
        self.assertAlmostEqual(d.policy.fist_countdown(d.t), 0.1, delta=0.05)
        d.hold(0, 0.2)
        self.assertEqual(d.policy.state, LOCKED)
        self.assertEqual(d.kinds()[-1], "fist_lock")
        self.assertAlmostEqual(d.events[-1].data["held_s"], 3.0, delta=0.05)
        self.assertIsNone(d.policy.until)
        d.hold(0, 5)                    # locked: a fist does nothing more
        self.assertEqual(d.kinds().count("fist_lock"), 1)

    def test_opening_the_hand_or_leaving_before_3_s_cancels(self):
        for release in (3, NO_HAND):
            d = Driver()
            d.hold(3, 2.5)
            d.hold(0, 2.0)
            self.assertEqual(d.countdown(), [3, 2])
            d.hold(release, 0.5)        # hand opens (3 fingers) or leaves the square
            self.assertEqual(d.kinds()[-1], "countdown_cancel")
            self.assertIsNone(d.policy.fist_countdown(d.t))
            d.hold(0, 2.0)              # a new fist starts over from 3, and 2 s isn't enough
            self.assertEqual(d.policy.state, UNLOCKED)
            self.assertEqual(d.countdown(), [3, 2, 3, 2])

    def test_a_flicker_shorter_than_the_gap_does_not_cancel(self):
        d = Driver()
        d.hold(3, 2.5)
        d.hold(0, 1.5)
        d.frames(3, 3, confidence=0.6)  # 0.1 s of an unsure frame
        d.hold(0, 1.6)
        self.assertNotIn("countdown_cancel", d.kinds())
        self.assertEqual(d.policy.state, LOCKED)

    def test_fists_in_the_first_2_s_after_unlocking_are_ignored(self):
        d = Driver()
        d.hold(0, 1.9)                  # curling the hand right after the last digit
        self.assertEqual(d.countdown(), [])
        self.assertIsNone(d.policy.fist_countdown(d.t))
        self.assertEqual(d.policy.state, UNLOCKED)
        d.hold(0, 2.9)                  # still a fist: the countdown starts only after the grace period
        self.assertEqual(d.policy.state, UNLOCKED)
        self.assertEqual(d.countdown(), [3, 2, 1])
        d.hold(0, 0.5)
        self.assertEqual(d.policy.state, LOCKED)
        self.assertGreater(d.now, 2.0 + 3.0)

    def test_backup_relock_after_2_minutes(self):
        d = Driver()
        self.assertEqual(d.policy.config.relock_s, 120)
        d.hold(3, 60)                   # an open hand never locks it
        d.t += 59.5
        d.events += d.policy.tick(d.t)
        self.assertEqual(d.policy.state, UNLOCKED)
        self.assertAlmostEqual(d.policy.seconds_left(d.t), 120 - d.now, places=6)
        d.hold(NO_HAND, 1)
        self.assertEqual(d.policy.state, LOCKED)
        self.assertEqual(d.kinds()[-1], "relocked")


class CommandTests(unittest.TestCase):
    def run_events(self, d):
        sent = []
        log = type("Log", (), {"write": lambda *a: None})()
        handle_events(d.events, d.t, lambda cmd, a=0, b=0: sent.append((cmd, a)), log,
                      {"send_digits": False, "relock_s": 120, "max_wrong": 3})
        return sent

    def test_countdown_cancel_and_lock_commands_and_sounds(self):
        d = Driver()
        d.hold(3, 2.5)
        d.hold(0, 1.5)
        d.hold(3, 0.5)
        d.hold(0, 3.2)
        sent = self.run_events(d)
        self.assertEqual(sent, [("countdown", 3), ("countdown", 2), ("countdown", 0),
                                ("countdown", 3), ("countdown", 2), ("countdown", 1), ("locked", 1)])
        played = [sounds.sound_for(cmd, a) for cmd, a in sent]
        self.assertEqual(played, ["beep.wav", "beep.wav", None, "beep.wav", "beep.wav", "beep.wav", "locked.wav"])
        self.assertIsNone(sounds.sound_for("locked", 0))  # startup / resync: silent

    def test_backup_relock_sends_locked(self):
        d = Driver()
        d.t += 121
        d.events += d.policy.tick(d.t)
        self.assertEqual(self.run_events(d), [("locked", 1)])


if __name__ == "__main__":
    unittest.main()
