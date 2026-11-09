"""Tests for the Mac lock app's event handling, attempt log and window, without a camera or broker."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "lock"), str(ROOT / "board" / "python")]

import numpy as np  # noqa: E402

import lock_app  # noqa: E402
import lock_protocol as proto  # noqa: E402
from common.lock_policy import LockPolicy, PolicyConfig  # noqa: E402

SECRET = "ab" * 32
CONFIG = {"send_digits": False, "max_wrong": 3}


class FakeLink:
    """Records what would be published, checking each command exactly as the board does."""

    def __init__(self):
        self.signer = proto.Signer(SECRET)
        self.sent, self.rejected, self.last = [], [], None
        self.connected, self.board = True, {"online": True, "state": "locked"}

    def send(self, cmd, a=0, b=0):
        cmd_dict, reason = proto.verify_command(SECRET, self.signer.sign(cmd, a, b), self.last)
        if cmd_dict is None:
            self.rejected.append((cmd, reason))
            return
        self.last = cmd_dict["ts_ms"]
        self.sent.append((cmd, a, b))


class AppTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log_path = Path(self.tmp.name) / "attempts.jsonl"
        self.log = lock_app.Logger(self.log_path)
        self.link = FakeLink()
        self.policy = LockPolicy(PolicyConfig(code=(2, 4, 1), stable_frames=3))
        self.now = 100.0

    def run_frames(self, value, frames=4, config=CONFIG):
        for _ in range(frames):
            self.now += 0.04
            lock_app.handle_events(self.policy.update(self.now, value, 0.95), self.now, self.link, self.log,
                                   config, self.policy)

    def press(self, config=CONFIG):
        lock_app.handle_events(self.policy.press(self.now, "board"), self.now, self.link, self.log, config, self.policy)

    def enter(self, digits, config=CONFIG):
        self.press(config)
        self.run_frames(0, config=config)
        for d in digits:
            self.run_frames(d, config=config)
            self.run_frames(0, config=config)

    def log_entries(self):
        return [json.loads(line) for line in self.log_path.read_text().splitlines()]

    def test_correct_code_sends_armed_digits_unlock(self):
        self.enter([2, 4, 1])
        self.assertEqual(self.link.sent, [("armed", 0, 0), ("digit", 1, 0), ("digit", 2, 0), ("digit", 3, 0),
                                          ("unlock", 0, 0)])
        self.assertEqual(self.link.rejected, [])  # the board would accept every one, in order
        entry = self.log_entries()[0]
        self.assertEqual((entry["result"], entry["digits"], entry["source"]), ("unlock", [2, 4, 1], "board"))

    def test_digits_sent_only_when_configured(self):
        self.enter([2, 4, 1], config={**CONFIG, "send_digits": True})
        self.assertEqual([s for s in self.link.sent if s[0] == "digit"], [("digit", 1, 2), ("digit", 2, 4), ("digit", 3, 1)])

    def test_third_wrong_code_logs_the_lockout(self):
        for _ in range(3):
            self.enter([1, 1, 1])
        self.assertEqual(self.link.sent[-2:], [("deny", 0, 0), ("lockout", 30, 0)])
        entries = self.log_entries()
        self.assertEqual([e["result"] for e in entries], ["deny"] * 3)
        self.assertNotIn("lockout_s", entries[1])
        self.assertEqual(entries[2]["lockout_s"], 30)
        # a press during the lockout resends the remaining lockout to the board
        self.press()
        self.assertEqual(self.link.sent[-1][0], "lockout")

    def test_timeout_is_sent_and_logged(self):
        self.press()
        self.now += 11
        lock_app.handle_events(self.policy.tick(self.now), self.now, self.link, self.log, CONFIG, self.policy)
        self.assertEqual(self.link.sent[-1], ("timeout", 0, 0))
        self.assertEqual(self.log_entries()[0]["result"], "timeout")

    def test_reading_line_tells_no_hand_from_fist(self):
        text = lambda *r: lock_app.reading_text(*r, 0.8)[0]
        self.assertEqual(text(None, 0.0, True), "NO HAND (paused)")
        self.assertEqual(text(0, 0.95, False), "0 (FIST)  95%")
        self.assertEqual(text(3, 0.9, False), "3 fingers  90%")
        self.assertEqual(text(1, 0.9, False), "1 finger  90%")
        self.assertIn("unsure", text(2, 0.5, False))

    def test_window_draws_in_every_state(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        for setup in (lambda: None, self.press, lambda: self.enter([2, 4, 1]),
                      lambda: [self.enter([1, 1, 1]) for _ in range(3)]):
            setup()
            for reading in ((None, 0.0, True), (0, 0.95, False), (3, 0.5, False)):
                screen = lock_app.draw(frame, (10, 10, 100, 100), self.policy, reading, 1.2, 6.0,
                                       self.link, False, 2.0, self.now)
                self.assertEqual(screen.shape, (480, 640 + lock_app.PANEL_WIDTH, 3))


if __name__ == "__main__":
    unittest.main()
