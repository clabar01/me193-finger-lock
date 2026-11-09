"""MQTT messages between the Mac (camera, model, lock policy) and AirFour (button, servo, displays).

Shared by both sides: the board app imports it directly, and the Mac adds board/python
to its path. Standard library only.

Topics (public broker, as in the earlier ME193 projects):
    <prefix>/button   board -> Mac   {"ts_ms": ...}                 arcade button pressed
    <prefix>/cmd      Mac -> board   {"cmd", "a", "b", "ts_ms", "sig"}
    <prefix>/board    board -> Mac   {"state", "online", "ts_ms"}   retained; "online": false is the will

Commands (a, b are integers):
    armed    show the 3 empty slots, LED3 blue
    digit    a = slot 1-3, b = the digit (0 unless the Mac sends real digits for debugging)
    unlock   servo to unlocked, LED3 green, relock after a seconds
    deny     flash an X 3 times, LED4 red
    timeout  back to locked
    lockout  a = seconds of lockout
    locked   back to locked now (the Mac relocks or resynchronizes); a = 1: announce it
             (a fist held 3 s, or the 2 min backup relock: "Locked" on the speaker)
    countdown  while unlocked, a = seconds (3, 2, 1) before a held fist locks; 0 = cancelled
    servo    a = angle 0-180, for tuning the latch (lock/servo_test.py); the mode doesn't change

The broker is public, so anyone can publish to these topics. Every command
carries an HMAC-SHA256 signature made with a secret shared by the Mac
(config/lock.json) and the board (lock_secret.txt in the app folder, never in
git). The board accepts a command only if the signature is right, the
timestamp is within MAX_AGE_MS of its own clock and newer than the last
command it accepted, so an old "unlock" can't be replayed. Button and state
messages are not signed: faking one can only start an attempt or confuse a
display, never open the lock. The code itself never leaves the Mac.
"""
import hashlib
import hmac
import json
import time

BROKER = ("broker.hivemq.com", 1883)
TOPIC_PREFIX = "ME193/ceci/lock"
MAX_AGE_MS = 15_000
COMMANDS = ("armed", "digit", "unlock", "deny", "timeout", "lockout", "locked", "countdown", "servo")


def topics(prefix=TOPIC_PREFIX):
    return {"button": f"{prefix}/button", "cmd": f"{prefix}/cmd", "board": f"{prefix}/board"}


def now_ms():
    return int(time.time() * 1000)


def _mac(secret, cmd, a, b, ts_ms):
    message = f"{cmd}|{int(a)}|{int(b)}|{int(ts_ms)}".encode()
    return hmac.new(bytes.fromhex(secret), message, hashlib.sha256).hexdigest()


def sign_command(secret, cmd, a=0, b=0, ts_ms=None):
    """A command payload (JSON string) for the cmd topic."""
    if cmd not in COMMANDS:
        raise ValueError(f"unknown command {cmd!r}")
    ts_ms = now_ms() if ts_ms is None else int(ts_ms)
    return json.dumps({"cmd": cmd, "a": int(a), "b": int(b), "ts_ms": ts_ms, "sig": _mac(secret, cmd, a, b, ts_ms)})


class Signer:
    """Signs commands with strictly increasing timestamps.

    The board rejects a timestamp that is not newer than the last one it accepted,
    and the Mac often sends two commands in the same millisecond (the last digit,
    then unlock). So each timestamp is at least 1 ms after the previous one.
    Use one Signer per program; two programs sending at once can still interleave
    out of order, so don't run lock_app.py and servo_test.py together.
    """

    def __init__(self, secret):
        self.secret = secret
        self.last_ts_ms = 0

    def sign(self, cmd, a=0, b=0):
        self.last_ts_ms = max(now_ms(), self.last_ts_ms + 1)
        return sign_command(self.secret, cmd, a, b, self.last_ts_ms)


def verify_command(secret, payload, last_ts_ms, now=None):
    """Check a cmd payload. Returns (command dict or None, reason)."""
    try:
        msg = json.loads(payload)
        cmd, a, b, ts_ms, sig = msg["cmd"], int(msg["a"]), int(msg["b"]), int(msg["ts_ms"]), str(msg["sig"])
    except (ValueError, KeyError, TypeError):
        return None, "malformed"
    if cmd not in COMMANDS:
        return None, f"unknown command {cmd!r}"
    if not hmac.compare_digest(sig, _mac(secret, cmd, a, b, ts_ms)):
        return None, "bad signature"
    now = now_ms() if now is None else now
    if abs(now - ts_ms) > MAX_AGE_MS:
        return None, f"stale: {(now - ts_ms) / 1000:+.1f} s from this clock"
    if last_ts_ms is not None and ts_ms <= last_ts_ms:
        return None, "replayed or out of order"
    return {"cmd": cmd, "a": a, "b": b, "ts_ms": ts_ms}, "ok"
