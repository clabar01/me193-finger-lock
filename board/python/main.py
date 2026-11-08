"""me193_lock (AirFour): picks the mode from board_config.json, written by board/deploy_board.sh.

    "local"  the whole lock runs here: webcam, ONNX model, no-hand check and lock policy,
             talking to the sketch through the Bridge, with a web page on port 7000
             (local_lock.py). Needs no Mac and no internet.
    "mac"    the Mac runs the camera, model and policy and sends signed commands over
             MQTT (mac_link.py, used with lock/lock_app.py on the Mac).

The sketch is the same in both modes.
"""
import json
from pathlib import Path

CONFIG = Path(__file__).with_name("board_config.json")
mode = json.loads(CONFIG.read_text()).get("mode", "mac") if CONFIG.exists() else "mac"

if mode == "local":
    import local_lock
    local_lock.run()
else:
    import mac_link
    mac_link.run()
