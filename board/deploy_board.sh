#!/usr/bin/env bash
# Deploy the board app (board/) to AirFour and start it.
#
#   board/deploy_board.sh --mode local           # whole lock on the board (no Mac); web page on :7000
#   board/deploy_board.sh --mode mac             # the Mac runs the lock (lock/lock_app.py) over MQTT
#   board/deploy_board.sh --mode local --sketch  # also copy the sketch (recompiled and reflashed)
#   BOARD=arduino@192.168.1.185 board/deploy_board.sh --mode local
#                                     # default BOARD is arduino@airfour.local
#
# Copies the Python side, the shared common/ modules, the ONNX model, the web page,
# and from config/lock.json the lock settings (with the code) and the MQTT signing
# secret. Those two are written readable by the owner only and never go into git.
# A ROI calibrated on the board (data/roi.json) is kept; config/roi.json is only
# the starting point. This does not change the board's default app. 'app start'
# fails with 'app "X" is running' while another app runs: stop that app first.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=""
SKETCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --sketch) SKETCH=1; shift ;;
    -h | --help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (use --help)" >&2; exit 2 ;;
  esac
done
[[ "$MODE" == "local" || "$MODE" == "mac" ]] || { echo "give --mode local or --mode mac" >&2; exit 2; }
[[ -f config/lock.json ]] || { echo "config/lock.json not found: run lock/lock_app.py once to create it" >&2; exit 1; }
[[ -f models/finger_cnn.onnx ]] || { echo "models/finger_cnn.onnx not found" >&2; exit 1; }

BOARD="${BOARD:-arduino@airfour.local}"
APP="ArduinoApps/me193_lock"

echo "deploying to $BOARD:$APP in $MODE mode"
ssh "$BOARD" "mkdir -p $APP/python/common $APP/python/model $APP/assets/libs $APP/data $APP/logs"
scp -q board/app.yaml "$BOARD:$APP/"
scp -q board/python/*.py board/python/requirements.txt "$BOARD:$APP/python/"
scp -q common/__init__.py common/preprocess.py common/no_hand.py common/lock_policy.py \
  common/lock_runtime.py common/red_square.py common/draw.py "$BOARD:$APP/python/common/"
scp -q models/finger_cnn.onnx "$BOARD:$APP/python/model/"
scp -q config/roi.json "$BOARD:$APP/python/roi_default.json"
scp -q board/assets/index.html board/assets/app.js board/assets/style.css "$BOARD:$APP/assets/"
scp -q board/assets/libs/socket.io.min.js "$BOARD:$APP/assets/libs/"
if [[ $SKETCH == 1 ]]; then
  scp -q -r board/sketch "$BOARD:$APP/"
fi
python3 -c 'import json; c = json.load(open("config/lock.json")); c.pop("secret"); print(json.dumps(c, indent=2))' |
  ssh "$BOARD" "umask 077; cat > $APP/python/lock_settings.json"
python3 -c 'import json; print(json.load(open("config/lock.json"))["secret"])' |
  ssh "$BOARD" "umask 077; cat > $APP/python/lock_secret.txt"
printf '{"mode": "%s"}\n' "$MODE" | ssh "$BOARD" "cat > $APP/python/board_config.json"
echo "board clock: $(ssh "$BOARD" date -u '+%H:%M:%S')   Mac clock: $(date -u '+%H:%M:%S') (UTC)"
echo "restarting me193_lock (stop + start; 'app restart' can fail with 'app is running')"
ssh "$BOARD" "timeout -k 5 120 arduino-app-cli app stop ~/$APP; timeout -k 5 420 arduino-app-cli app start ~/$APP"
