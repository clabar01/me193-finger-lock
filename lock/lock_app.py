"""The finger-count lock, Mac side: webcam, model, no-hand check, lock policy, MQTT to AirFour.

    python lock/lock_app.py
    python lock/lock_app.py --mask-digits      # dots instead of digits in the window (demo video)

Flow: the arcade button on AirFour (or B here) -> a fresh snapshot of the empty
square for the no-hand check -> digits are read (a fist separates them; an
empty square only pauses reading) until the policy unlocks,
denies, times out or locks out -> signed commands go to AirFour, which moves the
servo and updates its displays. Every attempt is logged to logs/lock_attempts.jsonl.

Settings, the code and the signing secret are in config/lock.json (not in git).
The first run creates it from config/lock.example.json with a random secret;
then run board/deploy_board.sh so the board gets the same secret.

Keys: B = button (backup for testing), + / - = no-hand threshold, Q = quit.
"""
import argparse
import json
import queue
import secrets
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "board" / "python")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import paho.mqtt.client as mqtt  # noqa: E402

import lock_protocol as proto  # noqa: E402
from common.camera import CameraError, lock_and_report, open_configured_camera, restore_auto  # noqa: E402
from common.draw import label  # noqa: E402
from common.lock_policy import LOCKED, LOCKOUT, READING, UNLOCKED, LockPolicy, frame_reading  # noqa: E402
from common.lock_runtime import AttemptLog, STATE_COLORS, policy_config, reading_text, softmax  # noqa: E402
from common.lock_runtime import handle_events as _handle_events  # noqa: E402
from common.no_hand import EmptySquareCheck  # noqa: E402
from common.preprocess import ROI_PATH, crop_and_resize, load_roi, preprocess, roi_to_pixels  # noqa: E402

CONFIG = ROOT / "config" / "lock.json"
EXAMPLE = ROOT / "config" / "lock.example.json"
LOG = ROOT / "logs" / "lock_attempts.jsonl"
MODEL = ROOT / "models" / "finger_cnn.onnx"
WINDOW = "finger lock"
PANEL_WIDTH = 400
THRESHOLD_STEP = 0.5


def load_config(path=CONFIG):
    """Read config/lock.json, creating it (with a random secret) on the first run."""
    if not path.exists():
        shutil.copy(EXAMPLE, path)
        config = json.loads(path.read_text())
        config["secret"] = secrets.token_hex(32)
        path.write_text(json.dumps(config, indent=2) + "\n")
        print(f"Created {path} with a new secret. Set your code in it, then run board/deploy_board.sh.")
    config = json.loads(path.read_text())
    try:
        bytes.fromhex(config["secret"])
    except ValueError:
        raise SystemExit(f"{path}: 'secret' is not hex; delete the file to make a new one")
    return config


Logger = AttemptLog  # the attempt log, logs/lock_attempts.jsonl by default


def handle_events(events, now, link, log, config, policy=None):
    """See common.lock_runtime.handle_events; here commands go out over signed MQTT."""
    return _handle_events(events, now, link.send, log, config)


class Link:
    """MQTT: signed commands out, button presses and board state in."""

    def __init__(self, secret):
        self.signer = proto.Signer(secret)
        self.topics = proto.topics()
        self.buttons = queue.Queue()
        self.connected = False
        self.board = {"online": False, "state": "unknown"}
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"mac-lock-{secrets.token_hex(3)}")
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = lambda *a, **k: setattr(self, "connected", False)
        self.client.on_message = self._on_message
        self.client.connect_async(*proto.BROKER, keepalive=30)
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        self.connected = True
        client.subscribe([(self.topics["button"], 1), (self.topics["board"], 1)])

    def _on_message(self, client, userdata, msg):
        if msg.topic == self.topics["button"]:
            self.buttons.put("board")
        elif msg.topic == self.topics["board"]:
            try:
                self.board = json.loads(msg.payload)
            except ValueError:
                pass

    def send(self, cmd, a=0, b=0):
        self.client.publish(self.topics["cmd"], self.signer.sign(cmd, a, b), qos=1)

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()


def draw(frame, box, policy, reading, diff, threshold, link, mask, ms, now):
    value, confidence, no_hand = reading
    state = policy.state
    color = STATE_COLORS[state]
    view = frame.copy()
    cv2.rectangle(view, box[:2], box[2:], color, 3)
    panel = np.full((view.shape[0], PANEL_WIDTH, 3), 35, np.uint8)

    left = policy.seconds_left(now)
    title = state.upper() + (f"  {left:.0f} s" if left is not None and state != LOCKED else "")
    label(panel, title, (16, 50), 1.1, color, 3)

    digits = policy.digits if state == READING else []
    slots = [("*" if mask else str(d)) for d in digits] + ["_"] * (3 - len(digits))
    label(panel, "code  " + "  ".join(slots), (16, 100), 1.0, (255, 255, 255), 2)
    if state == READING:
        hint = "make a fist to separate" if policy.need_clear else f"show digit {len(digits) + 1} of 3"
        label(panel, hint, (16, 134), 0.6, (255, 160, 40))
    elif state == LOCKED:
        label(panel, "press the button to start", (16, 134), 0.6)

    text, text_color = reading_text(value, confidence, no_hand, policy.config.min_confidence)
    label(panel, text, (16, 180), 0.7, text_color, 2)
    streak_value, streak, needed = policy.stability()
    bar = int(260 * min(streak, needed) / needed)
    cv2.rectangle(panel, (16, 194), (276, 212), (80, 80, 80), -1)
    cv2.rectangle(panel, (16, 194), (16 + bar, 212), (0, 220, 0) if streak >= needed else (180, 180, 180), -1)
    streak_name = {None: "-", 0: "fist"}.get(streak_value, streak_value)
    label(panel, f"{streak_name}: {min(streak, needed)}/{needed}", (286, 209), 0.5)

    if diff is None:
        label(panel, "no-hand: no snapshot yet", (16, 250), 0.55, (0, 200, 255))
    else:
        label(panel, f"no-hand difference {diff:.1f}  (threshold {threshold:.1f})", (16, 250), 0.55,
              (255, 200, 0) if no_hand else (255, 255, 255))
    label(panel, f"wrong in a row {policy.wrong_in_a_row}/{policy.config.max_wrong}", (16, 280), 0.55)

    board = link.board
    board_text = f"AirFour {board.get('state', '?')}" if board.get("online") else "AirFour offline"
    label(panel, f"MQTT {'connected' if link.connected else 'DISCONNECTED'}   {board_text}", (16, 320), 0.5,
          (0, 220, 0) if link.connected and board.get("online") else (0, 200, 255))
    label(panel, f"model {ms:.0f} ms   B button  +/- threshold  Q quit", (16, view.shape[0] - 14), 0.45)
    return np.hstack((view, panel))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the finger-count lock (Mac side).")
    parser.add_argument("--mask-digits", action="store_true", help="show * instead of the digits in the window")
    args = parser.parse_args(argv)

    import onnxruntime as ort

    config = load_config()
    if list(config["code"]) == [1, 2, 3]:
        print("Note: the code is still the example 1-2-3. Change it in config/lock.json.")
    policy = LockPolicy(policy_config(config))
    for path, hint in ((MODEL, "run training/train.py --final"), (ROI_PATH, "run capture/calibrate_roi.py")):
        if not path.exists():
            print(f"{path} not found: {hint}")
            return 1
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    mode = session.get_modelmeta().custom_metadata_map["preprocess_mode"]
    roi = load_roi()
    check = EmptySquareCheck(config["empty_threshold"])
    link = Link(config["secret"])
    log = Logger(LOG)

    try:
        cap, cam = open_configured_camera()
    except CameraError as e:
        print(e)
        link.close()
        return 1
    try:
        settings = lock_and_report(cap, cam)
        print(f"Camera: {settings['camera']}, exposure {settings['exposure']}; model mode {mode!r}; "
              f"MQTT {proto.BROKER[0]} {proto.TOPIC_PREFIX}; log {LOG.relative_to(ROOT)}")
        link.send("locked")  # start from the same state on both sides
        while True:
            ok, frame = cap.read()
            if not ok:
                print("The camera stopped delivering frames.")
                return 1
            now = time.monotonic()
            crop = crop_and_resize(frame, roi)

            presses = []
            while not link.buttons.empty():
                presses.append(link.buttons.get())
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("b"), ord("B")):
                presses.append("mac")
            events = []
            for source in presses:
                if policy.state not in (UNLOCKED, LOCKOUT):
                    check.set_background(crop)  # the square should be empty right after the press
                events += policy.press(now, source)

            start = time.perf_counter()
            probs = softmax(session.run(None, {input_name: preprocess(crop, mode=mode)[None]})[0][0])
            ms = 1000 * (time.perf_counter() - start)
            diff, no_hand = check.check(crop)
            value, confidence = frame_reading(probs, no_hand)
            events += policy.update(now, value, confidence)
            for line in handle_events(events, now, link, log, config, policy):
                print(f"{datetime.now():%H:%M:%S}  {line}")

            box = roi_to_pixels(roi, frame.shape[1], frame.shape[0])
            cv2.imshow(WINDOW, draw(frame, box, policy, (value, confidence, no_hand), diff, check.threshold,
                                    link, args.mask_digits, ms, now))
            if key in (ord("q"), ord("Q")):
                return 0
            if key in (ord("+"), ord("=")):
                check.threshold += THRESHOLD_STEP
            elif key in (ord("-"), ord("_")):
                check.threshold = max(0.0, check.threshold - THRESHOLD_STEP)
    finally:
        restore_auto(cap, cam.get("opened_index"))
        cap.release()
        cv2.destroyAllWindows()
        link.close()


if __name__ == "__main__":
    sys.exit(main())
