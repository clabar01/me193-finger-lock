"""Lock pieces shared by the Mac app (lock/lock_app.py) and the AirFour app (board/python/local_lock.py).

policy_config() builds the policy from the settings in config/lock.json, handle_events()
turns policy events into commands for the sketch and attempt-log lines, and
reading_text() is the one wording of a frame's reading, used in the Mac window
and on the board's web page. Standard library and NumPy only.
"""
import json
import time
from datetime import datetime

import numpy as np

from common.lock_policy import LOCKED, LOCKOUT, READING, UNLOCKED, PolicyConfig

STATE_COLORS = {LOCKED: (200, 200, 200), READING: (255, 160, 40), UNLOCKED: (0, 220, 0), LOCKOUT: (60, 60, 255)}


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


SETTING_KEYS = {
    "code", "stable_frames", "min_confidence", "timeout_s", "max_wrong", "lockout_s",
    "fist_lock_s", "fist_grace_s", "fist_gap_s", "fist_release_s", "empty_threshold", "send_digits", "secret",
}
# Anything that would make the lock relock on a timer. The only way to lock while unlocked is
# a fist held fist_lock_s; tests/test_no_timer_relock.py fails if one of these comes back.
TIMER_WORDS = ("relock", "auto_lock", "autolock", "unlock_s", "unlocked_s", "unlock_timeout")


def check_settings(config):
    """Raise ValueError for a timer-based relock setting or an unknown key (an old config copy)."""
    timers = sorted(k for k in config if any(w in k.lower() for w in TIMER_WORDS))
    if timers:
        raise ValueError(f"timer relock settings are not allowed (only a held fist locks): {timers}")
    unknown = sorted(set(config) - SETTING_KEYS)
    if unknown:
        raise ValueError(f"unknown settings (an old config?): {unknown}")
    policy_config(config)  # the code and values must also be valid


def policy_config(config):
    keys = ("stable_frames", "min_confidence", "timeout_s", "max_wrong", "lockout_s",
            "fist_lock_s", "fist_grace_s", "fist_gap_s", "fist_release_s")  # optional: missing ones keep PolicyConfig's default
    return PolicyConfig(code=tuple(config["code"]), **{k: config[k] for k in keys if k in config})


class AttemptLog:
    """Appends one JSON line per finished attempt; the policy's times are converted to wall clock."""

    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record, now, extra):
        wall = time.time()
        entry = {"started_at": datetime.fromtimestamp(wall - (now - record["started"])).isoformat(timespec="milliseconds"),
                 "result": record["result"], "source": record["source"],
                 "digits": [d["value"] for d in record["digits"]],
                 "confidences": [d["confidence"] for d in record["digits"]],
                 "digit_times_s": [d["t"] for d in record["digits"]],
                 "duration_s": record["duration_s"], "wrong_in_a_row": record["wrong_in_a_row"], **extra}
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry


ALIVE_PERIOD_S = 5.0  # the sketch's failsafe relocks after 30 s without a word (6 missed)


class Heartbeat:
    """Says when to send the sketch "alive", so its failsafe never fires while the app runs."""

    def __init__(self, period=ALIVE_PERIOD_S):
        self.period, self.last = period, None

    def due(self, now):
        if self.last is None or now - self.last >= self.period:
            self.last = now
            return True
        return False


class BoardWatch:
    """Spots the board going from unlocked to locked on its own (the sketch's failsafe)."""

    def __init__(self):
        self.last = None

    def relocked(self, name):
        relocked = self.last == "unlocked" and name == "locked"
        self.last = name
        return relocked


def handle_events(events, now, send, log, config):
    """Send each policy event to the board with send(cmd, a, b) and log finished attempts.

    The same on the Mac (send = signed MQTT) and on AirFour (send = Bridge to the sketch).
    Returns text lines for the console.
    """
    lines, records, extra = [], [], {}
    for e in events:
        if e.kind == "armed":
            send("armed")
            lines.append(f"armed by the {e.data['source']} button: snapshot taken")
        elif e.kind == "digit":
            send("digit", e.data["index"], e.data["value"] if config["send_digits"] else 0)
            lines.append(f"digit {e.data['index']}: {e.data['value']} ({e.data['confidence']:.0%})")
        elif e.kind in ("unlock", "deny", "timeout", "restarted"):
            if e.kind == "unlock":
                send("unlock")
            elif e.kind in ("deny", "timeout"):
                send(e.kind)
            records.append(e.data["attempt"])
            lines.append(f"attempt: {e.kind}")
        elif e.kind == "lockout":
            send("lockout", int(e.data["seconds"]))
            extra["lockout_s"] = e.data["seconds"]
            lines.append(f"{config['max_wrong']} wrong codes: locked out for {e.data['seconds']:.0f} s")
        elif e.kind == "ignored":
            left = e.data.get("seconds_left") or 0
            if e.data["reason"] == LOCKOUT:
                send("lockout", max(1, round(left)))  # the board may have reacted to the press: put it back
            lines.append(f"button ignored: {e.data['reason']} ({left:.0f} s left)")
        elif e.kind == "cleared":
            lines.append("fist: the next digit can come")
        elif e.kind == "countdown":
            send("countdown", e.data["seconds"])
            lines.append(f"fist: locking in {e.data['seconds']}")
        elif e.kind == "countdown_cancel":
            send("countdown", 0)
            lines.append("fist released: countdown cancelled")
        elif e.kind == "fist_lock":
            send("locked", 1)  # a=1: announce it ("Locked" on the speaker)
            lines.append(f"locked by a fist held {e.data['held_s']} s")
        elif e.kind == "board_relocked":
            lines.append("the board relocked by itself (failsafe: it heard nothing from this app)")
        elif e.kind == "lockout_over":
            lines.append("lockout over")
    for i, record in enumerate(records):  # a lockout belongs to the last attempt of the frame
        log.write(record, now, extra if i == len(records) - 1 else {})
    return lines


def unlocked_hint(policy, now):
    """The hint line while unlocked: the fist-lock countdown, the grace period, or how to lock."""
    left = policy.fist_countdown(now)
    if left is not None:
        return f"LOCKING IN {max(1, -int(-left // 1))}  (open your hand to cancel)"
    if policy.unlocked_at is not None and now - policy.unlocked_at < policy.config.fist_grace_s:
        return "unlocked"
    if not policy.fist_armed:
        return "open your hand or take it away, then hold a fist 3 s to lock"
    return f"hold a fist {policy.config.fist_lock_s:.0f} s to lock"


CYAN, AMBER, WHITE, GREEN = (255, 200, 0), (0, 200, 255), (255, 255, 255), (0, 220, 0)


def reading_text(value, confidence, no_hand, min_confidence):
    """The window's reading line and its color: no hand, unsure, 0 (fist) or a finger count."""
    if no_hand:
        return "NO HAND (paused)", CYAN
    if confidence < min_confidence:
        return f"unsure ({value}?, {confidence:.0%})", AMBER
    if value == 0:
        return f"0 (FIST)  {confidence:.0%}", GREEN
    return f"{value} finger{'s' if value > 1 else ''}  {confidence:.0%}", WHITE
