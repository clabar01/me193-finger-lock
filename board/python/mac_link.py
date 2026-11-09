"""Mac mode (board_config.json "mode": "mac"): MQTT link between the sketch and the Mac.

    sketch Bridge.notify("button")      -> publish <prefix>/button
    sketch Bridge.notify("lock_state")  -> publish <prefix>/board (retained)
    <prefix>/cmd from the Mac           -> verify signature and freshness -> Bridge.call("lock_cmd")

The Mac runs the camera, the model and the lock policy (lock/lock_app.py); this
board never knows the code. Commands are signed with the secret in
lock_secret.txt next to this file (copied by board/deploy_board.sh, never in
git); without it every command is rejected. See lock_protocol.py.
"""
import json
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt
from arduino.app_utils import App, Bridge, Logger

import lock_protocol as proto
from sounds import Sounds

logger = Logger("me193_lock")
sounds = Sounds(log=logger.warning)  # Bluetooth speaker; never blocks, see sounds.py
TOPICS = proto.topics()
SECRET_PATH = Path(__file__).with_name("lock_secret.txt")

secret = SECRET_PATH.read_text().strip() if SECRET_PATH.exists() else None
state = {"state": "locked", "last_ts_ms": None}
lock = threading.Lock()


def publish_state(client):
    payload = json.dumps({"state": state["state"], "online": True, "ts_ms": proto.now_ms()})
    client.publish(TOPICS["board"], payload, qos=1, retain=True)


def on_connect(client, userdata, flags, reason_code, properties=None):
    logger.info(f"MQTT connected to {proto.BROKER[0]} ({reason_code}); board clock {time.strftime('%H:%M:%S')}")
    client.subscribe(TOPICS["cmd"], qos=1)
    publish_state(client)


def on_message(client, userdata, msg):
    if secret is None:
        logger.warning("command ignored: no lock_secret.txt on the board (run board/deploy_board.sh)")
        return
    with lock:
        cmd, reason = proto.verify_command(secret, msg.payload.decode(errors="replace"), state["last_ts_ms"])
        if cmd is None:
            logger.warning(f"command rejected: {reason}")
            return
        state["last_ts_ms"] = cmd["ts_ms"]
    lag = proto.now_ms() - cmd["ts_ms"]
    logger.info(f"cmd {cmd['cmd']} a={cmd['a']} b={cmd['b']} ({lag} ms after the Mac sent it)")
    Bridge.call("lock_cmd", cmd["cmd"], cmd["a"], cmd["b"])
    sounds.on_command(cmd["cmd"], cmd["a"], cmd["b"])


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"airfour-lock-{proto.now_ms() % 100000}")
client.will_set(TOPICS["board"], json.dumps({"state": "unknown", "online": False}), qos=1, retain=True)
client.on_connect = on_connect
client.on_message = on_message


def button(ms):
    logger.info("button pressed")
    client.publish(TOPICS["button"], json.dumps({"ts_ms": proto.now_ms()}), qos=1)


def lock_state(name):
    state["state"] = name
    logger.info(f"state: {name}")
    publish_state(client)


def lock_status(text):
    logger.info(f"sketch: {text}")


def idle_loop():
    time.sleep(1)


def run():
    """Start the MQTT link and the app loop (returns when the app stops)."""
    if secret is None:
        logger.warning(f"{SECRET_PATH.name} missing: commands from the Mac will be rejected")
    Bridge.provide("button", button)
    Bridge.provide("lock_state", lock_state)
    Bridge.provide("lock_status", lock_status)
    client.connect_async(*proto.BROKER, keepalive=30)
    client.loop_start()
    logger.info(f"me193_lock started; topics under {proto.TOPIC_PREFIX}")

    try:
        App.run(user_loop=idle_loop)
    finally:
        client.publish(TOPICS["board"], json.dumps({"state": "unknown", "online": False}), qos=1, retain=True)
        client.loop_stop()
