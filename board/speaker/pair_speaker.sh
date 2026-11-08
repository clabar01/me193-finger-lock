#!/usr/bin/env bash
# Pair the lock's Bluetooth speaker with AirFour (run on the Mac; works over SSH).
#
#   board/speaker/pair_speaker.sh                 # looks for a device named "JBL Flip 5"
#   NAME="JBL Flip" board/speaker/pair_speaker.sh
#   BOARD=arduino@192.168.1.185 board/speaker/pair_speaker.sh
#
# First put the speaker in pairing mode (JBL Flip 5: switch it on, press the Bluetooth
# button; it beeps and the Bluetooth light blinks fast). Then this script:
#   1. scans up to 40 s for the speaker, 2. pairs, trusts and connects it,
#   3. saves its address in ~/ArduinoApps/me193_lock/data/speaker.json (the lock app plays
#      sounds to it), 4. installs and starts airfour-speaker.service (reconnect + keep awake).
# Trusted + the service = it reconnects after reboots without running this again.
set -euo pipefail
cd "$(dirname "$0")"
BOARD="${BOARD:-arduino@airfour.local}"
NAME="${NAME:-JBL Flip 5}"
APP="ArduinoApps/me193_lock"

scp -q airfour-speaker.sh "$BOARD:/tmp/airfour-speaker.sh"
scp -q airfour-speaker.service "$BOARD:/tmp/airfour-speaker.service"
ssh "$BOARD" NAME="'$NAME'" APP="$APP" bash -s <<'EOF'
set -euo pipefail
bluetoothctl power on >/dev/null
bluetoothctl pairable on >/dev/null
echo "scanning for \"$NAME\" (up to 40 s)..."
MAC=""
for i in $(seq 1 8); do
  bluetoothctl --timeout 5 scan on >/dev/null 2>&1 || true
  MAC="$(bluetoothctl devices | grep -F "$NAME" | head -1 | awk '{print $2}')" || true
  [[ -n "$MAC" ]] && break
done
if [[ -z "$MAC" ]]; then
  echo "not found. Is the speaker in pairing mode (Bluetooth light blinking fast)? Seen:"
  bluetoothctl devices
  exit 1
fi
echo "found $NAME at $MAC"
if ! bluetoothctl info "$MAC" | grep -q "Paired: yes"; then
  bluetoothctl --agent NoInputNoOutput --timeout 20 pair "$MAC" || true
fi
bluetoothctl info "$MAC" | grep -q "Paired: yes" || { echo "pairing failed: run this again with the speaker in pairing mode"; exit 1; }
bluetoothctl trust "$MAC" >/dev/null
bluetoothctl --timeout 15 connect "$MAC" || true
sleep 3
bluetoothctl info "$MAC" | grep -E "Name|Paired|Trusted|Connected"
mkdir -p ~/$APP/data
printf '{"mac": "%s", "name": "%s"}\n' "$MAC" "$NAME" > ~/$APP/data/speaker.json
mkdir -p ~/.local/bin ~/.config/systemd/user
install -m 755 /tmp/airfour-speaker.sh ~/.local/bin/airfour-speaker.sh
install -m 644 /tmp/airfour-speaker.service ~/.config/systemd/user/airfour-speaker.service
systemctl --user daemon-reload
systemctl --user enable --now airfour-speaker.service >/dev/null 2>&1
systemctl --user restart airfour-speaker.service
echo "airfour-speaker.service: $(systemctl --user is-active airfour-speaker.service), enabled at boot"
wpctl status | sed -n '/Sinks:/,/Sources:/p' | head -8
EOF
