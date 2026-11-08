#!/usr/bin/env bash
# Deploy the board app (board/) to AirFour and start it.
#
#   board/deploy_board.sh             # Python side and the signing secret
#   board/deploy_board.sh --sketch    # also copies the sketch (recompiled and reflashed on start)
#   BOARD=arduino@192.168.1.173 board/deploy_board.sh   # at home (default is the
#                                     # "airfour" SSH alias, the school Wi-Fi address)
#
# The signing secret comes from config/lock.json (run lock/lock_app.py once to
# create it) and goes to python/lock_secret.txt on the board, never into git.
# This does not change the board's default app. 'app start' fails with
# 'app "X" is running' while another app runs (after a reboot, the default app):
# stop that app first.
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-}" in
  "" | --sketch) ;;
  -h | --help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown option: $1 (use --sketch or --help)" >&2; exit 2 ;;
esac

BOARD="${BOARD:-airfour}"
APP="~/ArduinoApps/me193_lock"
SECRET=$(python3 -c 'import json,sys; print(json.load(open("config/lock.json"))["secret"])' 2>/dev/null) || {
  echo "config/lock.json not found: run lock/lock_app.py once to create it" >&2; exit 1; }

files=(board/app.yaml board/python/main.py board/python/lock_protocol.py board/python/requirements.txt)
if [[ "${1:-}" == "--sketch" ]]; then
  files+=(board/sketch)
fi

echo "copying ${files[*]} to $BOARD:ArduinoApps/me193_lock/"
ssh "$BOARD" "mkdir -p $APP/python"
scp -q board/app.yaml "$BOARD:ArduinoApps/me193_lock/"
scp -q board/python/main.py board/python/lock_protocol.py board/python/requirements.txt "$BOARD:ArduinoApps/me193_lock/python/"
if [[ "${1:-}" == "--sketch" ]]; then
  scp -q -r board/sketch "$BOARD:ArduinoApps/me193_lock/"
fi
printf '%s\n' "$SECRET" | ssh "$BOARD" "umask 077; cat > $APP/python/lock_secret.txt"
echo "board clock: $(ssh "$BOARD" date -u '+%H:%M:%S')   Mac clock: $(date -u '+%H:%M:%S') (UTC)   (commands older than 15 s are rejected)"
echo "restarting me193_lock (stop + start; 'app restart' can fail with 'app is running')"
ssh "$BOARD" "timeout -k 5 120 arduino-app-cli app stop $APP; timeout -k 5 420 arduino-app-cli app start $APP"
