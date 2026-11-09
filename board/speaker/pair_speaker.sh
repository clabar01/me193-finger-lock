#!/usr/bin/env bash
# Pair the lock's Bluetooth speaker with AirFour (run on the Mac; works over SSH).
#
#   board/speaker/pair_speaker.sh "JBL Go 4"
#   BOARD=arduino@192.168.1.185 board/speaker/pair_speaker.sh "JBL Flip 5"
#
# First put the speaker in pairing mode (JBL: switch it on, press the Bluetooth button;
# the Bluetooth light blinks fast). Then this script:
#   1. scans up to 40 s for a device with that name, 2. pairs, trusts and connects it,
#   3. removes the previous lock speaker (from data/speaker.json), so AirFour stops
#      trying to reconnect to it,
#   4. saves the new address in ~/ArduinoApps/me193_lock/data/speaker.json (the lock app
#      and the reconnect service play to it), and installs and restarts
#      airfour-speaker.service (reconnect + keep awake),
#   5. keeps Bluetooth outputs from suspending when idle (81-bluez-no-suspend.conf), makes it
#      the default audio output and plays "Access granted" on it.
# Trusted + the service = it reconnects after reboots without running this again.
# If it never shows up as an audio output, run board/speaker/fix_bluetooth_audio.sh once.
set -euo pipefail
cd "$(dirname "$0")"
[[ $# -eq 1 && -n "$1" ]] || { echo "usage: $0 \"<speaker name>\"   e.g. $0 \"JBL Go 4\"" >&2; exit 2; }
NAME="$1"
BOARD="${BOARD:-arduino@airfour.local}"

scp -q airfour-speaker.sh "$BOARD:/tmp/airfour-speaker.sh"
scp -q airfour-speaker.service "$BOARD:/tmp/airfour-speaker.service"
scp -q 81-bluez-no-suspend.conf "$BOARD:/tmp/81-bluez-no-suspend.conf"
ssh "$BOARD" "bash -s -- $(printf %q "$NAME")" <<'EOF'
set -euo pipefail
NAME="$1"
APP=~/ArduinoApps/me193_lock
SAVED="$APP/data/speaker.json"
OLD=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["mac"])' "$SAVED" 2>/dev/null || true)

bluetoothctl power on >/dev/null
bluetoothctl pairable on >/dev/null
echo "scanning for \"$NAME\" (up to 40 s)..."
MAC=""
for i in $(seq 1 8); do
  bluetoothctl --timeout 5 scan on >/dev/null 2>&1 || true
  MAC="$(bluetoothctl devices | awk -v n="$NAME" '{m=$2; $1=$2=""; sub(/^  /,""); if ($0==n) {print m; exit}}')"
  [[ -n "$MAC" ]] && break
done
if [[ -z "$MAC" ]]; then
  echo "not found. Is the speaker in pairing mode (Bluetooth light blinking fast)? Named devices seen:"
  bluetoothctl devices | grep -Ev '^Device ([0-9A-F]{2}:){5}[0-9A-F]{2} ([0-9A-F]{2}-){5}[0-9A-F]{2}$' | head -20
  exit 1
fi
echo "found $NAME at $MAC"
if ! bluetoothctl info "$MAC" | grep -q "Paired: yes"; then
  bluetoothctl --agent NoInputNoOutput --timeout 20 pair "$MAC" 2>&1 | grep -E "Pairing successful|Failed|rror" || true
fi
bluetoothctl info "$MAC" | grep -q "Paired: yes" || { echo "pairing failed: run this again with the speaker in pairing mode"; exit 1; }
bluetoothctl trust "$MAC" >/dev/null

if [[ -n "$OLD" && "$OLD" != "$MAC" ]]; then
  echo "removing the previous speaker $OLD ($(bluetoothctl info "$OLD" 2>/dev/null | sed -n 's/^\tName: //p'))"
  bluetoothctl disconnect "$OLD" >/dev/null 2>&1 || true
  bluetoothctl remove "$OLD" >/dev/null 2>&1 || true
fi

mkdir -p "$APP/data"
printf '{"mac": "%s", "name": "%s"}\n' "$MAC" "$NAME" > "$SAVED"
mkdir -p ~/.local/bin ~/.config/systemd/user
install -m 755 /tmp/airfour-speaker.sh ~/.local/bin/airfour-speaker.sh
install -m 644 /tmp/airfour-speaker.service ~/.config/systemd/user/airfour-speaker.service
systemctl --user daemon-reload
systemctl --user enable airfour-speaker.service >/dev/null 2>&1
systemctl --user restart airfour-speaker.service

# Keep the Bluetooth output from suspending when idle (81-bluez-no-suspend.conf): a suspended
# output closes the stream, and the speaker drops the start of the next sound (the first beep).
WPCONF=~/.config/wireplumber/wireplumber.conf.d
mkdir -p "$WPCONF"
if ! cmp -s /tmp/81-bluez-no-suspend.conf "$WPCONF/81-bluez-no-suspend.conf"; then
  install -m 644 /tmp/81-bluez-no-suspend.conf "$WPCONF/81-bluez-no-suspend.conf"
  bluetoothctl disconnect "$MAC" >/dev/null 2>&1 || true
  systemctl --user restart wireplumber
  sleep 4
fi

bluetoothctl --timeout 15 connect "$MAC" >/dev/null 2>&1 || true
bluetoothctl info "$MAC" | grep -E "Name|Paired|Trusted|Connected"
echo "airfour-speaker.service: $(systemctl --user is-active airfour-speaker.service), enabled at boot"

NODE="bluez_output.${MAC//:/_}.1"
ID=""
for i in $(seq 1 15); do
  ID=$(pw-dump 2>/dev/null | python3 -c '
import json, sys
for o in json.load(sys.stdin):
    if o.get("info", {}).get("props", {}).get("node.name") == sys.argv[1]:
        print(o["id"]); break' "$NODE" || true)
  [[ -n "$ID" ]] && break
  sleep 1
done
if [[ -z "$ID" ]]; then
  echo "connected, but not an audio output yet: run board/speaker/fix_bluetooth_audio.sh once"
  exit 1
fi
wpctl set-default "$ID"
wpctl set-volume "$ID" 1.0
wpctl status | sed -n '/^Audio/,/Sources:/p' | grep -A4 "Sinks:"
echo "playing 'Access granted' on $NAME"
XDG_RUNTIME_DIR=/run/user/$(id -u) pw-play --target "$NODE" "$APP/python/sounds/granted.wav"
echo "done"
EOF
