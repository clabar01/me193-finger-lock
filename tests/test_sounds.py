"""Tests for board/python/sounds.py: the right sound per command, and a missing speaker never breaks the lock."""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "board" / "python"))

import sounds  # noqa: E402


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not cond():
        time.sleep(0.02)
    return cond()


class SoundsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        for name in sounds.FOR_COMMAND.values():
            (self.dir / name).write_bytes(b"")
        self.speaker = self.dir / "speaker.json"
        self.log = []

    def make(self, player):
        return sounds.Sounds(log=self.log.append, sound_dir=self.dir, speaker_file=self.speaker, player=player)

    def test_target_from_speaker_file(self):
        self.assertIsNone(sounds.speaker_target(self.speaker))
        self.speaker.write_text(json.dumps({"mac": "f8:df:15:aa:bb:01"}))
        self.assertEqual(sounds.speaker_target(self.speaker), "bluez_output.F8_DF_15_AA_BB_01.1")
        self.speaker.write_text("not json")
        self.assertIsNone(sounds.speaker_target(self.speaker))

    def test_commands_map_to_sounds_and_target(self):
        s = self.make("true")
        self.speaker.write_text(json.dumps({"mac": "F8:DF:15:AA:BB:01"}))
        self.assertEqual(s.command("beep.wav"),
                         ["true", "--target", "bluez_output.F8_DF_15_AA_BB_01.1", str(self.dir / "beep.wav")])
        self.assertEqual(sounds.FOR_COMMAND, {"digit": "beep.wav", "unlock": "granted.wav", "deny": "intruder.wav"})

    def test_failing_or_missing_player_never_raises_or_blocks(self):
        for player in ("false", "/no/such/player"):
            s = self.make(player)
            t = time.monotonic()
            for cmd in ("armed", "digit", "digit", "digit", "unlock", "deny", "timeout", "lockout"):
                s.on_command(cmd, 1, 2)  # must return at once
            self.assertLess(time.monotonic() - t, 0.1)
            self.assertTrue(wait_for(lambda: len(self.log) >= 3), self.log)
            self.assertIn("not played", self.log[0])
            self.log.clear()

    def test_queue_drops_extra_sounds(self):
        s = self.make("sleep")  # "sleep <path>" fails fast, but the queue fills before the worker runs
        for _ in range(50):
            s.play("beep.wav")
        self.assertLessEqual(s.queue.qsize(), 4)


if __name__ == "__main__":
    unittest.main()
