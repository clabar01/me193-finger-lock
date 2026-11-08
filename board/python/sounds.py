"""Lock sounds on the Bluetooth speaker (JBL Flip 5), played through the host's PipeWire.

    digit accepted -> beep.wav     unlock -> granted.wav ("Access granted")
    wrong code     -> intruder.wav (alarm + "Intruder detected")

The App Lab container already has the host's PipeWire socket (/run/user/1000/pipewire-0)
and pw-play, so the app plays the files itself. If data/speaker.json holds the speaker's
address (written by board/speaker/pair_speaker.sh), sounds go to that speaker; otherwise
they go to the default output.

This never blocks the lock and never raises: sounds are queued to a worker thread, and if
the speaker is off or disconnected, pw-play fails or plays elsewhere, and the sound is
dropped. The lock works the same with or without sound.
"""
import json
import os
import queue
import subprocess
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOUND_DIR = HERE / "sounds"
SPEAKER_FILE = HERE.parent / "data" / "speaker.json"
FOR_COMMAND = {"digit": "beep.wav", "unlock": "granted.wav", "deny": "intruder.wav"}
PLAY_TIMEOUT_S = 8


def speaker_target(path=SPEAKER_FILE):
    """PipeWire node name of the paired speaker, or None (use the default output)."""
    try:
        mac = json.loads(Path(path).read_text())["mac"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return "bluez_output." + mac.strip().upper().replace(":", "_") + ".1"


class Sounds:
    def __init__(self, log=print, sound_dir=SOUND_DIR, speaker_file=SPEAKER_FILE, player="pw-play"):
        self.log, self.sound_dir, self.speaker_file, self.player = log, Path(sound_dir), speaker_file, player
        self.queue = queue.Queue(maxsize=4)
        self.env = dict(os.environ)
        self.env.setdefault("XDG_RUNTIME_DIR", "/run/user/1000")
        threading.Thread(target=self._worker, daemon=True, name="sounds").start()

    def on_command(self, cmd, a=0, b=0):
        """Called with every command sent to the sketch; plays the matching sound, if any."""
        name = FOR_COMMAND.get(cmd)
        if name:
            self.play(name)

    def play(self, name):
        try:
            self.queue.put_nowait(name)
        except queue.Full:  # several sounds already waiting: drop this one
            pass

    def command(self, name):
        args = [self.player]
        target = speaker_target(self.speaker_file)
        if target:
            args += ["--target", target]
        return args + [str(self.sound_dir / name)]

    def _worker(self):
        while True:
            name = self.queue.get()
            if not (self.sound_dir / name).exists():
                self.log(f"sound {name} missing")
                continue
            try:
                r = subprocess.run(self.command(name), env=self.env, timeout=PLAY_TIMEOUT_S,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if r.returncode:
                    self.log(f"sound {name} not played: {r.stderr.decode(errors='replace').strip()[:200]}")
            except (OSError, subprocess.SubprocessError) as e:
                self.log(f"sound {name} not played: {e}")
