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


def policy_config(config):
    keys = ("stable_frames", "min_confidence", "timeout_s", "max_wrong", "lockout_s", "relock_s")
    return PolicyConfig(code=tuple(config["code"]), **{k: config[k] for k in keys})


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
                send("unlock", int(config["relock_s"]))
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
        elif e.kind in ("relocked", "lockout_over"):
            lines.append(e.kind.replace("_", " "))
    for i, record in enumerate(records):  # a lockout belongs to the last attempt of the frame
        log.write(record, now, extra if i == len(records) - 1 else {})
    return lines


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
