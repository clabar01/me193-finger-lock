"""Stand-in for AirFour, to test the Mac side without the board.

    python lock/fake_board.py          # Enter = press the arcade button, Ctrl-C = quit

Prints every command the Mac sends after checking its signature exactly as the
board does, and publishes the board's state like the real one (including
the end of a lockout, and the failsafe relock after 30 s with no command).
"""
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "board" / "python")]

import paho.mqtt.client as mqtt  # noqa: E402

import lock_protocol as proto  # noqa: E402

TOPICS = proto.topics()


def main():
    secret = json.loads((ROOT / "config" / "lock.json").read_text())["secret"]
    last = {"ts_ms": None, "heard": time.monotonic(), "state": "locked"}
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"fake-board-{proto.now_ms() % 100000}")

    def set_state(name):
        last["state"] = name
        client.publish(TOPICS["board"], json.dumps({"state": name, "online": True, "ts_ms": proto.now_ms()}),
                       qos=1, retain=True)
        print(f"  board state: {name}")

    def later(seconds, name):
        threading.Timer(seconds, set_state, args=(name,)).start()

    def on_connect(c, userdata, flags, reason_code, properties=None):
        print(f"fake board connected to {proto.BROKER[0]}; Enter = button")
        c.subscribe(TOPICS["cmd"], qos=1)
        set_state("locked")

    def on_message(c, userdata, msg):
        cmd, reason = proto.verify_command(secret, msg.payload.decode(errors="replace"), last["ts_ms"])
        if cmd is None:
            print(f"REJECTED: {reason}")
            return
        last["ts_ms"], last["heard"] = cmd["ts_ms"], time.monotonic()
        name = cmd["cmd"]
        if name == "alive":
            return
        print(f"cmd {name} a={cmd['a']} b={cmd['b']}")
        if name == "armed":
            set_state("reading")
        elif name == "unlock":
            set_state("unlocked")
        elif name == "deny":
            set_state("denied")
            later(1.5, "locked")
        elif name == "lockout":
            set_state("lockout")
            later(cmd["a"], "locked")
        elif name in ("timeout", "locked"):
            set_state("locked")

    client.will_set(TOPICS["board"], json.dumps({"state": "unknown", "online": False}), qos=1, retain=True)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(*proto.BROKER, keepalive=30)
    client.loop_start()

    def failsafe():  # like the sketch: unlocked + no command for 30 s -> locked
        while True:
            time.sleep(1)
            if last["state"] == "unlocked" and time.monotonic() - last["heard"] >= 30:
                print("  failsafe: no word from the Mac for 30 s")
                set_state("locked")

    threading.Thread(target=failsafe, daemon=True).start()
    try:
        for _ in sys.stdin:
            client.publish(TOPICS["button"], json.dumps({"ts_ms": proto.now_ms()}), qos=1)
            print("button pressed")
    except KeyboardInterrupt:
        pass
    finally:
        client.publish(TOPICS["board"], json.dumps({"state": "unknown", "online": False}), qos=1, retain=True)
        time.sleep(0.5)
        client.loop_stop()


if __name__ == "__main__":
    main()
