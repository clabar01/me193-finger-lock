"""The lock's rules, with no camera, model or network code, so they can be explained and unit tested.

Feed it one reading per camera frame and the button presses; it returns events
for the app to act on (send to the board, log, draw).

    policy = LockPolicy(PolicyConfig(code=(2, 4, 1)))
    events = policy.press(now, source="board")          # arcade button (or B on the Mac)
    value, confidence = frame_reading(probs, no_hand)   # "no hand" is no reading (None)
    events = policy.update(now, value, confidence)       # once per frame
    events = policy.tick(now)                            # timers, if no frame arrived

Rules:
  * The code is 3 digits from 1 to 5 (config/lock.json).
  * A reading is stable when the same value comes N frames in a row (stable_frames),
    each with confidence of at least min_confidence. A frame below the threshold
    breaks the streak.
  * "No hand" (the square matches the empty snapshot, common/no_hand.py) is no
    reading at all, whatever the CNN says: never a digit, never a separator. It
    pauses reading and resets the stability count.
  * 0 means a closed fist: the CNN predicting 0 while a hand is in the square.
    It is the only separator.
  * The first stable 1-5 after the button press is the first digit. After that,
    a digit counts only once a stable fist has come since the previous digit.
    So 2 then 2 needs 2, fist, 2; taking the hand away between them is not enough.
  * timeout_s with no progress (attempt start or accepted digit): back to locked.
    A timeout is not a wrong code.
  * After the last digit: unlock if it matches, otherwise deny. max_wrong wrong
    codes in a row: lockout for lockout_s; button presses are ignored until it ends.
    A correct code resets the count.
  * Unlocked lasts relock_s, then the lock is locked again (the board also relocks itself).
  * A button press while reading restarts the attempt.
"""
from dataclasses import dataclass, field

LOCKED, READING, UNLOCKED, LOCKOUT = "locked", "reading", "unlocked", "lockout"


@dataclass(frozen=True)
class PolicyConfig:
    code: tuple
    stable_frames: int = 10
    min_confidence: float = 0.8
    timeout_s: float = 10.0
    max_wrong: int = 3
    lockout_s: float = 30.0
    relock_s: float = 10.0

    def __post_init__(self):
        code = tuple(int(d) for d in self.code)
        if len(code) != 3 or any(d not in range(1, 6) for d in code):
            raise ValueError(f"the code must be 3 digits from 1 to 5, got {self.code}")
        object.__setattr__(self, "code", code)
        if self.stable_frames < 1:
            raise ValueError("stable_frames must be at least 1")


@dataclass
class Event:
    kind: str  # armed, cleared, digit, unlock, deny, lockout, lockout_over, timeout, relocked, restarted, ignored
    data: dict = field(default_factory=dict)


NO_HAND = None  # the value frame_reading gives for an empty square: no reading


def frame_reading(probs, no_hand):
    """(value, confidence) for one frame: (NO_HAND, 0.0) when the square is empty, else the CNN's answer."""
    if no_hand:
        return NO_HAND, 0.0
    best = int(max(range(len(probs)), key=lambda i: probs[i]))
    return best, float(probs[best])


class LockPolicy:
    def __init__(self, config):
        self.config = config
        self.state = LOCKED
        self.wrong_in_a_row = 0
        self.until = None  # end of the unlocked or lockout period
        self.attempt = None
        self._reset_streak()

    # ----- inputs -----

    def press(self, now, source="button"):
        """The button was pressed. Returns events (armed, or ignored with a reason)."""
        events = self.tick(now)
        if self.state in (UNLOCKED, LOCKOUT):
            return events + [Event("ignored", {"reason": self.state, "seconds_left": self.seconds_left(now)})]
        if self.state == READING:
            events.append(self._finish(now, "restarted"))
        self.state = READING
        self.digits = []
        self.need_clear = False  # the first digit needs no fist before it
        self.last_progress = now
        self.attempt = {"started": now, "source": source, "digits": [], "result": None}
        self._reset_streak()
        return events + [Event("armed", {"source": source})]

    def update(self, now, value, confidence):
        """One camera frame's reading: value 0 (fist) to 5, or None for no hand / no reading. Returns events."""
        events = self.tick(now)
        if self.state != READING:
            return events
        if value is None or confidence < self.config.min_confidence:
            self._reset_streak()
            return events
        if value == self.streak_value:
            self.streak_len += 1
            self.streak_confidence += confidence
        else:
            self.streak_value, self.streak_len, self.streak_confidence = value, 1, confidence
        if self.streak_len != self.config.stable_frames:
            return events  # not stable yet, or already handled when it became stable

        mean_confidence = self.streak_confidence / self.streak_len
        if value == 0:
            if self.need_clear:
                self.need_clear = False
                events.append(Event("cleared"))
            return events
        if self.need_clear:
            return events  # a digit is still showing, or a new one came without a 0 between
        self.digits.append(value)
        self.need_clear = True
        self.last_progress = now
        digit = {"value": value, "confidence": round(mean_confidence, 4),
                 "t": round(now - self.attempt["started"], 3)}
        self.attempt["digits"].append(digit)
        events.append(Event("digit", {"index": len(self.digits), **digit}))
        if len(self.digits) == len(self.config.code):
            events += self._judge(now)
        return events

    def tick(self, now):
        """Timers: attempt timeout, end of unlocked period, end of lockout."""
        if self.state == READING and now - self.last_progress >= self.config.timeout_s:
            self.state = LOCKED
            return [self._finish(now, "timeout")]
        if self.state == UNLOCKED and now >= self.until:
            self.state, self.until = LOCKED, None
            return [Event("relocked")]
        if self.state == LOCKOUT and now >= self.until:
            self.state, self.until = LOCKED, None
            self.wrong_in_a_row = 0
            return [Event("lockout_over")]
        return []

    # ----- for display -----

    def seconds_left(self, now):
        """Seconds until timeout (reading), relock (unlocked) or the end of the lockout; else None."""
        if self.state == READING:
            return max(0.0, self.config.timeout_s - (now - self.last_progress))
        if self.until is not None:
            return max(0.0, self.until - now)
        return None

    def stability(self):
        """(value, frames so far, frames needed) of the current streak."""
        return self.streak_value, self.streak_len, self.config.stable_frames

    # ----- internals -----

    def _reset_streak(self):
        self.streak_value, self.streak_len, self.streak_confidence = None, 0, 0.0

    def _judge(self, now):
        entered = tuple(self.digits)
        if entered == self.config.code:
            self.wrong_in_a_row = 0
            self.state, self.until = UNLOCKED, now + self.config.relock_s
            return [self._finish(now, "unlock")]
        self.wrong_in_a_row += 1
        events = [self._finish(now, "deny")]
        if self.wrong_in_a_row >= self.config.max_wrong:
            self.state, self.until = LOCKOUT, now + self.config.lockout_s
            events.append(Event("lockout", {"seconds": self.config.lockout_s}))
        else:
            self.state = LOCKED
        return events

    def _finish(self, now, result):
        """Close the current attempt; the event carries the record to log."""
        self.attempt.update({"result": result, "ended": now, "wrong_in_a_row": self.wrong_in_a_row,
                             "duration_s": round(now - self.attempt["started"], 3)})
        record, self.attempt = self.attempt, None
        self._reset_streak()
        return Event(result, {"attempt": record})
