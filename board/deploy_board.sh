#!/usr/bin/env bash
# Deploy the board app (board/) to the Uno Q and start it.
#
#   board/deploy_board.sh             # Python side only
#   board/deploy_board.sh --sketch    # also copies the sketch
#   BOARD=arduino@192.168.1.173 board/deploy_board.sh   # at home (default is the
#                                     # "airfour" SSH alias, the school Wi-Fi address)
#
# Either way 'app start' recompiles and reflashes the MCU. This does not change
# the board's default app. 'app start' fails with 'app "X" is running' while
# another app runs (after a reboot, the default app): stop that app first.
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-}" in
  "" | --sketch) ;;
  -h | --help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown option: $1 (use --sketch or --help)" >&2; exit 2 ;;
esac

BOARD="${BOARD:-airfour}"
APP="~/ArduinoApps/me193_lock"

files=(board/app.yaml board/python)
if [[ "${1:-}" == "--sketch" ]]; then
  files+=(board/sketch)
fi

echo "copying ${files[*]} to $BOARD:ArduinoApps/me193_lock/"
ssh "$BOARD" "mkdir -p $APP"
scp -q -r "${files[@]}" "$BOARD:ArduinoApps/me193_lock/"
echo "restarting me193_lock (stop + start; 'app restart' can fail with 'app is running')"
ssh "$BOARD" "timeout -k 5 120 arduino-app-cli app stop $APP; timeout -k 5 420 arduino-app-cli app start $APP"
