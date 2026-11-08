"""Tests for the lock policy (common/lock_policy.py) and the signed MQTT commands (board/python/lock_protocol.py)."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "board" / "python")]

import lock_protocol as proto  # noqa: E402
from common.lock_policy import (LOCKED, LOCKOUT, NO_HAND, READING, UNLOCKED, LockPolicy,  # noqa: E402
                                PolicyConfig, frame_reading)

FPS = 30
N = 10


class Driver:
    """Feeds the policy frames at 30 fps and collects events."""

    def __init__(self, **config):
        self.policy = LockPolicy(PolicyConfig(code=(2, 4, 1), stable_frames=N, **config))
        self.now = 0.0
        self.events = []

    def press(self, source="board"):
        self.events += self.policy.press(self.now, source)

    def show(self, value, frames=N + 2, confidence=0.95):
        for _ in range(frames):
            self.now += 1 / FPS
            self.events += self.policy.update(self.now, value, confidence)

    def wait(self, seconds):
        self.now += seconds
        self.events += self.policy.tick(self.now)

    def enter(self, digits):
        for d in digits:
            self.show(d)
            self.show(0)

    def kinds(self):
        return [e.kind for e in self.events]

    def digits(self):
        return [e.data["value"] for e in self.events if e.kind == "digit"]


class PolicyTests(unittest.TestCase):
    def test_correct_code_unlocks_then_relocks(self):
        d = Driver()
        d.press()
        d.show(0)
        d.enter([2, 4, 1])
        self.assertEqual(d.digits(), [2, 4, 1])
        self.assertIn("unlock", d.kinds())
        self.assertEqual(d.policy.state, UNLOCKED)
        d.wait(10)
        self.assertEqual(d.policy.state, LOCKED)
        self.assertEqual(d.kinds()[-1], "relocked")

    def test_wrong_code_denies(self):
        d = Driver()
        d.press()
        d.show(0)
        d.enter([2, 4, 2])
        self.assertIn("deny", d.kinds())
        self.assertEqual(d.policy.state, LOCKED)
        self.assertEqual(d.policy.wrong_in_a_row, 1)

    def test_first_digit_needs_no_fist(self):
        d = Driver()
        d.press()
        d.show(None)  # empty square right after the press
        d.show(2)
        self.assertEqual(d.digits(), [2])
        self.assertNotIn("cleared", d.kinds())

    def test_digits_after_the_first_need_a_fist(self):
        d = Driver()
        d.press()
        d.show(2)
        d.show(4)  # straight from 2 to 4 with no fist: ignored
        self.assertEqual(d.digits(), [2])
        d.show(0)  # fist
        d.show(4)
        self.assertEqual(d.digits(), [2, 4])

    def test_repeated_digit_needs_zero_between(self):
        d = Driver()
        d.press()
        d.show(0)
        d.show(2, frames=3 * N)  # held a long time: still one digit
        self.assertEqual(d.digits(), [2])
        d.show(3)                 # a different digit without clearing: ignored
        self.assertEqual(d.digits(), [2])
        d.show(0)
        d.show(2)
        self.assertEqual(d.digits(), [2, 2])

    def test_unstable_or_unsure_frames_do_not_count(self):
        d = Driver()
        d.press()
        d.show(0)
        d.show(2, frames=N - 1)
        d.show(3, frames=1)       # streak broken one frame short
        d.show(2, frames=N - 1)
        self.assertEqual(d.digits(), [])
        d.show(2, frames=2 * N, confidence=0.79)  # below min_confidence
        self.assertEqual(d.digits(), [])
        d.show(2, frames=N, confidence=0.8)
        self.assertEqual(d.digits(), [2])

    def test_frame_reading_no_hand_is_no_reading(self):
        probs = [0.97, 0.01, 0.01, 0.01, 0.0, 0.0]  # the CNN says 0
        self.assertEqual(frame_reading(probs, no_hand=True), (NO_HAND, 0.0))
        self.assertEqual(frame_reading(probs, no_hand=False), (0, 0.97))

    def test_no_hand_is_not_a_separator(self):
        d = Driver()
        d.press()
        d.show(2)
        d.show(NO_HAND, frames=3 * N)  # hand taken away
        d.show(2)                       # same digit again: not counted, no fist came between
        self.assertEqual(d.digits(), [2])
        self.assertNotIn("cleared", d.kinds())
        d.show(0)
        d.show(2)
        self.assertEqual(d.digits(), [2, 2])

    def test_no_hand_never_becomes_a_digit_or_a_fist(self):
        d = Driver()
        d.press()
        for _ in range(5 * N):  # even if the CNN would have said 0 or 3
            d.now += 1 / FPS
            d.events += d.policy.update(d.now, *frame_reading([0.0, 0.0, 0.0, 0.99, 0.01, 0.0], no_hand=True))
        self.assertEqual(d.kinds(), ["armed"])

    def test_no_hand_pauses_and_resets_the_count(self):
        d = Driver()
        d.press()
        d.show(2, frames=N - 1)
        d.show(NO_HAND, frames=1)
        d.show(2, frames=N - 1)  # 2 * (N - 1) frames of 2, but never N in a row
        self.assertEqual(d.digits(), [])
        d.show(2, frames=1)
        self.assertEqual(d.digits(), [2])

    def test_timeout_without_progress(self):
        d = Driver()
        d.press()
        d.show(0)
        d.show(2)
        d.wait(9.9)
        self.assertEqual(d.policy.state, READING)
        d.wait(0.2)
        self.assertEqual(d.policy.state, LOCKED)
        self.assertEqual(d.kinds()[-1], "timeout")
        self.assertEqual(d.policy.wrong_in_a_row, 0)  # a timeout is not a wrong code

    def test_three_wrong_codes_lock_out_for_30_s(self):
        d = Driver()
        for _ in range(3):
            d.press()
            d.show(0)
            d.enter([1, 1, 1])
        self.assertEqual(d.kinds().count("deny"), 3)
        self.assertIn("lockout", d.kinds())
        self.assertEqual(d.policy.state, LOCKOUT)
        d.press()
        self.assertEqual(d.events[-1].kind, "ignored")
        d.wait(29.0)
        self.assertEqual(d.policy.state, LOCKOUT)
        d.wait(1.1)
        self.assertEqual(d.policy.state, LOCKED)
        self.assertEqual(d.policy.wrong_in_a_row, 0)
        d.press()
        d.show(0)
        d.enter([2, 4, 1])
        self.assertEqual(d.policy.state, UNLOCKED)

    def test_correct_code_resets_wrong_count(self):
        d = Driver()
        for code in ([1, 1, 1], [1, 1, 1], [2, 4, 1]):
            d.press()
            d.show(0)
            d.enter(code)
        self.assertEqual(d.policy.wrong_in_a_row, 0)
        self.assertNotIn("lockout", d.kinds())

    def test_press_while_reading_restarts(self):
        d = Driver()
        d.press()
        d.show(0)
        d.show(2)
        d.press()
        self.assertIn("restarted", d.kinds())
        d.show(0)
        d.enter([2, 4, 1])
        self.assertEqual(d.policy.state, UNLOCKED)

    def test_press_while_unlocked_is_ignored(self):
        d = Driver()
        d.press()
        d.show(0)
        d.enter([2, 4, 1])
        d.press()
        self.assertEqual(d.events[-1].kind, "ignored")
        self.assertEqual(d.policy.state, UNLOCKED)

    def test_attempt_record_for_the_log(self):
        d = Driver()
        d.press(source="mac")
        d.show(0)
        d.enter([2, 4, 2])
        record = next(e.data["attempt"] for e in d.events if e.kind == "deny")
        self.assertEqual(record["source"], "mac")
        self.assertEqual([x["value"] for x in record["digits"]], [2, 4, 2])
        self.assertTrue(all(abs(x["confidence"] - 0.95) < 1e-6 for x in record["digits"]))
        self.assertEqual(record["result"], "deny")
        json.dumps(record)  # must be loggable

    def test_bad_code_rejected(self):
        for code in ((1, 2), (1, 2, 6), (0, 1, 2)):
            with self.assertRaises(ValueError):
                PolicyConfig(code=code)


class ProtocolTests(unittest.TestCase):
    SECRET = "00" * 31 + "01"

    def test_signed_command_verifies(self):
        payload = proto.sign_command(self.SECRET, "digit", 2, 0, ts_ms=1_000_000)
        cmd, reason = proto.verify_command(self.SECRET, payload, None, now=1_000_500)
        self.assertEqual(reason, "ok")
        self.assertEqual((cmd["cmd"], cmd["a"], cmd["b"]), ("digit", 2, 0))

    def test_wrong_secret_or_tampering_rejected(self):
        payload = proto.sign_command(self.SECRET, "deny", ts_ms=1_000_000)
        self.assertEqual(proto.verify_command("ff" * 32, payload, None, now=1_000_000)[1], "bad signature")
        forged = json.loads(payload)
        forged["cmd"] = "unlock"
        self.assertEqual(proto.verify_command(self.SECRET, json.dumps(forged), None, now=1_000_000)[1], "bad signature")

    def test_replay_and_stale_rejected(self):
        payload = proto.sign_command(self.SECRET, "unlock", 10, ts_ms=1_000_000)
        self.assertIsNone(proto.verify_command(self.SECRET, payload, 1_000_000, now=1_000_100)[0])
        self.assertIn("stale", proto.verify_command(self.SECRET, payload, None, now=1_000_000 + 60_000)[1])

    def test_same_millisecond_commands_all_accepted_via_signer(self):
        # The bug seen on the board: digit 3 and unlock in the same millisecond, the unlock rejected.
        same = [proto.sign_command(self.SECRET, c, ts_ms=1_000_000) for c in ("digit", "unlock")]
        first, _ = proto.verify_command(self.SECRET, same[0], None, now=1_000_000)
        self.assertEqual(proto.verify_command(self.SECRET, same[1], first["ts_ms"], now=1_000_000)[1],
                         "replayed or out of order")
        signer, last = proto.Signer(self.SECRET), None
        for c in ("digit", "unlock", "locked", "servo"):
            cmd, reason = proto.verify_command(self.SECRET, signer.sign(c), last)
            self.assertEqual(reason, "ok", c)
            last = cmd["ts_ms"]

    def test_malformed_and_unknown_rejected(self):
        self.assertEqual(proto.verify_command(self.SECRET, "not json", None)[1], "malformed")
        with self.assertRaises(ValueError):
            proto.sign_command(self.SECRET, "open_sesame")


if __name__ == "__main__":
    unittest.main()
