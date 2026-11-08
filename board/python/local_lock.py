"""Local mode (board_config.json "mode": "local"): the whole lock on AirFour, no Mac, no internet.

Every frame: webcam -> ROI crop -> no-hand check (snapshot on each button press)
-> ONNX model -> lock policy (common/lock_policy.py) -> commands to the sketch over
the Bridge (the same commands the Mac sends over MQTT, handled the same way by
common/lock_runtime.py). The sketch, its timing and its displays are unchanged.

Web page (App Lab WebUI brick, port 7000): live camera, prediction, confidence, lock
state and timing, a backup button, ROI calibration, and a live accuracy test.

Files in the app folder (/app is ~/ArduinoApps/me193_lock on the board):
    python/model/finger_cnn.onnx   the model trained on the Mac (never trained here)
    python/lock_settings.json      the code and policy settings (from config/lock.json)
    data/roi.json                  the ROI; the Calibrate button rewrites it
    logs/lock_attempts.jsonl       every attempt;  logs/live_eval.jsonl  accuracy-test trials
"""
import json
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge, Logger

from board_camera import CameraError, open_camera
from sounds import Sounds
from common.lock_policy import LockPolicy, READING, frame_reading
from common.lock_runtime import AttemptLog, STATE_COLORS, handle_events, policy_config, reading_text, softmax
from common.no_hand import EmptySquareCheck
from common.preprocess import crop_and_resize, load_roi, preprocess, roi_to_pixels
from common.red_square import box_to_roi, find_red_square

HERE = Path(__file__).resolve().parent
APP = HERE.parent
MODEL = HERE / "model" / "finger_cnn.onnx"
SETTINGS = HERE / "lock_settings.json"
ROI_FILE = APP / "data" / "roi.json"
ROI_DEFAULT = HERE / "roi_default.json"
LOGS = APP / "logs"
# onnxruntime on AirFour (lock/benchmark.py, 240 crops, nothing else running): the default
# (4 threads, spinning while idle) took 21.7 ms/frame but 2.6 cores at 30 fps; 3 threads
# without spinning took 22.0 ms and 1.7 cores. 1 thread: 47 ms, too slow for 30 fps.
ORT_THREADS = 3
STATUS_PERIOD_S = 0.2
STREAM_FPS = 8
EVAL_TIMEOUT_S = 10.0

logger = Logger("me193_lock")
sounds = None  # started in run(); see sounds.py


def bridge_send(cmd, a=0, b=0):
    Bridge.call("lock_cmd", cmd, int(a), int(b))
    if sounds:  # after the sketch has the command, so sound never delays the servo or displays
        sounds.on_command(cmd, a, b)


def hex_color(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


class Usage:
    """CPU (this process, % of one core; whole board, % of all 4 cores) and memory, sampled each second."""

    def __init__(self):
        self.hz = os.sysconf("SC_CLK_TCK")
        self.last = None
        self.values = {"cpu_process": None, "cpu_board": None, "mem_mb": None}

    def sample(self):
        now = time.monotonic()
        with open("/proc/self/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        ticks = int(fields[11]) + int(fields[12])
        with open("/proc/stat") as f:
            cpu = [int(x) for x in f.readline().split()[1:]]
        idle, total = cpu[3] + cpu[4], sum(cpu)
        if self.last:
            t0, ticks0, idle0, total0 = self.last
            self.values["cpu_process"] = round(100 * (ticks - ticks0) / self.hz / (now - t0), 1)
            self.values["cpu_board"] = round(100 * (1 - (idle - idle0) / max(total - total0, 1)), 1)
        self.last = (now, ticks, idle, total)
        with open("/proc/self/status") as f:
            rss = next(int(line.split()[1]) for line in f if line.startswith("VmRSS"))
        self.values["mem_mb"] = round(rss / 1024, 1)


class LiveEval:
    """Accuracy test: pick the expected label, show it; the next stable reading (what the
    lock would accept: stable_frames in a row above min_confidence) is one trial."""

    def __init__(self, stable_frames, min_confidence, path):
        self.n, self.min_conf, self.path = stable_frames, min_confidence, path
        self.expected, self.trials = None, []
        if path.exists():
            self.trials = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self._reset()

    def _reset(self):
        self.value, self.count, self.conf_sum = None, 0, 0.0

    def start(self, expected):
        self.expected, self.started = expected, time.monotonic()
        self._reset()

    def feed(self, now, value, confidence, no_hand):
        if self.expected is None:
            return
        if now - self.started > EVAL_TIMEOUT_S:
            return self._record(None, 0.0, now)
        if no_hand or confidence < self.min_conf:
            self._reset()
            return
        if value == self.value:
            self.count, self.conf_sum = self.count + 1, self.conf_sum + confidence
        else:
            self.value, self.count, self.conf_sum = value, 1, confidence
        if self.count == self.n:
            self._record(value, self.conf_sum / self.count, now)

    def _record(self, predicted, confidence, now):
        trial = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "expected": self.expected, "predicted": predicted,
                 "confidence": round(confidence, 4), "seconds": round(now - self.started, 2)}
        self.trials.append(trial)
        with open(self.path, "a") as f:
            f.write(json.dumps(trial) + "\n")
        self.expected = None
        self._reset()

    def reset(self):
        self.trials, self.expected = [], None
        self.path.unlink(missing_ok=True)

    def summary(self):
        per = {str(k): {"n": 0, "correct": 0} for k in range(6)}
        for t in self.trials:
            per[str(t["expected"])]["n"] += 1
            per[str(t["expected"])]["correct"] += t["predicted"] == t["expected"]
        n = sum(p["n"] for p in per.values())
        correct = sum(p["correct"] for p in per.values())
        return {"expected": self.expected, "per_label": per, "n": n, "correct": correct,
                "last": self.trials[-5:][::-1]}


class LocalLock:
    def __init__(self):
        settings = json.loads(SETTINGS.read_text())
        self.settings = settings
        self.policy = LockPolicy(policy_config(settings))
        self.check = EmptySquareCheck(settings["empty_threshold"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = ORT_THREADS
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(MODEL), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.mode = self.session.get_modelmeta().custom_metadata_map["preprocess_mode"]
        self.roi = load_roi(ROI_FILE if ROI_FILE.exists() else ROI_DEFAULT)
        self.log = AttemptLog(LOGS / "lock_attempts.jsonl")
        self.eval = LiveEval(settings["stable_frames"], settings["min_confidence"], LOGS / "live_eval.jsonl")
        self.presses = queue.Queue()
        self.lock = threading.Lock()
        self.frame = None
        self.jpeg = None
        self.stream_clients = 0
        self.frame_times = deque(maxlen=60)
        self.infer_ms = deque(maxlen=60)
        self.process_ms = deque(maxlen=60)
        self.usage = Usage()
        self.last_status = self.last_usage = 0.0
        self.reading = (None, 0.0, False, None)
        self.cap, self.camera = open_camera(log=logger.warning)
        logger.info(f"camera: {self.camera}")
        logger.info(f"model {MODEL.name} (mode {self.mode!r}), ROI {self.roi}")

    # ----- inputs from other threads -----

    def button(self, source):
        self.presses.put(source)

    def calibrate(self):
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
        if frame is None:
            return {"ok": False, "message": "no camera frame yet"}
        box = find_red_square(frame)
        if box is None:
            return {"ok": False, "message": "no red square found: the whole square, border included, "
                                           "must be in view and empty"}
        roi = box_to_roi(box, frame.shape[1], frame.shape[0])
        ROI_FILE.parent.mkdir(exist_ok=True)
        ROI_FILE.write_text(json.dumps(roi, indent=2) + "\n")
        with self.lock:
            self.roi = roi
        logger.info(f"ROI calibrated: {roi}")
        return {"ok": True, "message": f"ROI saved: {box[2] - box[0]} px square at ({box[0]}, {box[1]})", "roi": roi}

    # ----- the frame loop (App.run's user_loop) -----

    def step(self):
        ok, frame = self.cap.read()
        if not ok:
            logger.error("camera read failed")
            time.sleep(0.5)
            return
        now = time.monotonic()
        self.frame_times.append(now)
        t0 = time.perf_counter()
        with self.lock:
            roi = self.roi
            self.frame = frame
        crop = crop_and_resize(frame, roi)

        events = []
        while not self.presses.empty():
            source = self.presses.get()
            if self.policy.state not in ("unlocked", "lockout"):
                self.check.set_background(crop)
            events += self.policy.press(now, source)

        t1 = time.perf_counter()
        x = preprocess(crop, mode=self.mode)[None]
        probs = softmax(self.session.run(None, {self.input_name: x})[0][0])
        self.infer_ms.append(1000 * (time.perf_counter() - t1))
        diff, no_hand = self.check.check(crop)
        value, confidence = frame_reading(probs, no_hand)
        events += self.policy.update(now, value, confidence)
        for line in handle_events(events, now, bridge_send, self.log, self.settings):
            logger.info(line)
        self.eval.feed(now, value, confidence, no_hand)
        self.reading = (value if not no_hand else None, confidence, no_hand, diff)
        self.process_ms.append(1000 * (time.perf_counter() - t0))

        if self.stream_clients and (not self.jpeg or now - self.jpeg[0] >= 1 / STREAM_FPS):
            self._encode(frame, roi, now)
        if now - self.last_usage >= 1.0:
            self.last_usage = now
            self.usage.sample()
        if now - self.last_status >= STATUS_PERIOD_S:
            self.last_status = now
            ui.send_message("status", self.status(now))

    def _encode(self, frame, roi, now):
        view = frame.copy()
        box = roi_to_pixels(roi, view.shape[1], view.shape[0])
        cv2.rectangle(view, box[:2], box[2:], STATE_COLORS[self.policy.state], 3)
        ok, jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.lock:
                self.jpeg = (now, jpeg.tobytes())

    def status(self, now):
        value, confidence, no_hand, diff = self.reading
        text, color = reading_text(value, confidence, no_hand, self.policy.config.min_confidence)
        digits = self.policy.digits if self.policy.state == READING else []
        shown = [str(d) if self.settings.get("send_digits") else "●" for d in digits]
        streak_value, streak, needed = self.policy.stability()
        times = list(self.frame_times)
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
        hint = ""
        if self.policy.state == READING:
            hint = "make a fist to separate" if self.policy.need_clear else f"show digit {len(digits) + 1} of 3"
        left = self.policy.seconds_left(now)
        return {
            "state": self.policy.state, "state_color": hex_color(STATE_COLORS[self.policy.state]),
            "seconds_left": None if left is None else round(left, 1), "slots": shown + ["_"] * (3 - len(shown)),
            "hint": hint, "reading": text, "reading_color": hex_color(color),
            "stability": {"value": "fist" if streak_value == 0 else streak_value, "count": min(streak, needed), "needed": needed},
            "no_hand_diff": None if diff is None else round(diff, 1), "threshold": self.check.threshold,
            "wrong_in_a_row": self.policy.wrong_in_a_row, "max_wrong": self.policy.config.max_wrong,
            "fps": round(fps, 1), "infer_ms": round(float(np.mean(self.infer_ms)), 2) if self.infer_ms else None,
            "process_ms": round(float(np.mean(self.process_ms)), 2) if self.process_ms else None,
            **self.usage.values, "camera": self.camera, "eval": self.eval.summary(),
        }

    def stream(self):
        """MJPEG of the latest annotated frame, for <img src="/stream">."""
        with self.lock:
            self.stream_clients += 1
        try:
            last = None
            while True:
                with self.lock:
                    jpeg = self.jpeg
                if jpeg and jpeg[0] != last:
                    last = jpeg[0]
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg[1] + b"\r\n"
                time.sleep(1 / (2 * STREAM_FPS))
        finally:
            with self.lock:
                self.stream_clients -= 1


ui = WebUI()


def run():
    global sounds
    sounds = Sounds(log=logger.warning)
    try:
        lock = LocalLock()
    except CameraError as e:
        logger.error(f"camera: {e}")
        raise

    from fastapi.responses import StreamingResponse

    def api_status():
        return lock.status(time.monotonic())

    def api_press():
        lock.button("web")
        return {"ok": True}

    def api_calibrate():
        return lock.calibrate()

    def api_eval(label: int):
        if label not in range(6):
            return {"ok": False, "message": "label must be 0-5"}
        lock.eval.start(label)
        return {"ok": True}

    def api_eval_reset():
        lock.eval.reset()
        return {"ok": True}

    ui.expose_api("GET", "/stream", lambda: StreamingResponse(lock.stream(), media_type="multipart/x-mixed-replace; boundary=frame"))
    ui.expose_api("GET", "/api/status", api_status)
    ui.expose_api("POST", "/api/press", api_press)
    ui.expose_api("POST", "/api/calibrate", api_calibrate)
    ui.expose_api("POST", "/api/eval/{label}", api_eval)
    ui.expose_api("POST", "/api/eval_reset", api_eval_reset)

    def button(ms):
        logger.info("button pressed")
        lock.button("board")

    def lock_state(name):
        logger.info(f"board state: {name}")

    def lock_status(text):
        logger.info(f"sketch: {text}")

    Bridge.provide("button", button)
    Bridge.provide("lock_state", lock_state)
    Bridge.provide("lock_status", lock_status)
    bridge_send("locked")
    logger.info("me193_lock running in local mode (no Mac); web page on port 7000")
    try:
        App.run(user_loop=lock.step)
    finally:
        lock.cap.release()
