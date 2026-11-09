"""Make the lock's sound files (run on the Mac; the WAVs are small and go into git).

    python board/speaker/make_sounds.py            # writes board/python/sounds/*.wav
    python board/speaker/make_sounds.py --voice Daniel

beep.wav      short 1.2 kHz beep for each accepted digit
granted.wav   rising two-note chime + "Access granted"   (macOS text-to-speech, `say`)
intruder.wav  three falling alarm tones + "Intruder detected"
locked.wav    falling two-note chime + "Locked"
keepalive.wav 1 s of a very quiet 40 Hz tone (about -45 dBFS) that board/speaker/
              airfour-speaker.sh plays every 4 minutes so the JBL's auto-off doesn't trigger

Every file starts with 0.4 s of silence: a Bluetooth speaker that was idle needs a
moment to start the stream, and would otherwise clip the start of the sound.
"""
import argparse
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

RATE = 44100
OUT = Path(__file__).resolve().parents[1] / "python" / "sounds"
LEAD_IN_S = 0.4


def tone(freq, seconds, volume=0.5, fade_s=0.01):
    t = np.arange(int(RATE * seconds)) / RATE
    x = volume * np.sin(2 * np.pi * freq * t)
    n = min(int(RATE * fade_s), len(x) // 2)
    ramp = np.linspace(0, 1, n)
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def silence(seconds):
    return np.zeros(int(RATE * seconds))


def speech(text, voice):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.wav"
        subprocess.run(["say", "-v", voice, "-o", str(path), f"--data-format=LEI16@{RATE}", text], check=True)
        with wave.open(str(path)) as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32768
            return data.reshape(-1, w.getnchannels()).mean(axis=1)


def write(name, x, normalize=True):
    x = np.concatenate([silence(LEAD_IN_S), x, silence(0.1)])
    if normalize:
        x = 0.9 * x / max(np.abs(x).max(), 1e-9)
    OUT.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT / name), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    print(f"wrote {OUT / name} ({len(x) / RATE:.1f} s)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--voice", default="Samantha", help="a macOS voice (say -v '?')")
    args = parser.parse_args(argv)
    write("beep.wav", tone(1200, 0.12))
    write("granted.wav", np.concatenate([tone(660, 0.15), tone(990, 0.25), silence(0.15),
                                         speech("Access granted", args.voice)]))
    alarm = np.concatenate([np.concatenate([tone(f, 0.18), silence(0.05)]) for f in (1000, 800, 1000, 800)])
    write("intruder.wav", np.concatenate([alarm, silence(0.15), speech("Intruder detected", args.voice)]))
    write("locked.wav", np.concatenate([tone(990, 0.15), tone(660, 0.25), silence(0.15),
                                        speech("Locked", args.voice)]))
    write("keepalive.wav", tone(40, 1.0, volume=10 ** (-45 / 20), fade_s=0.2), normalize=False)


if __name__ == "__main__":
    main()
