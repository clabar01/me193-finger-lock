#!/usr/bin/env bash
# Make the paired Bluetooth speaker an audio output on AirFour (run once, on the Mac).
#
#   board/speaker/fix_bluetooth_audio.sh     # asks for the board's sudo password once
#
# Why it's needed: AirFour boots to the lightdm login screen, which runs its own audio
# server (PipeWire + WirePlumber as the "lightdm" user) on the active seat. That one
# grabs Bluetooth audio first, so the speaker connects but never appears as an output
# for the "arduino" user, whose audio the lock app uses. This script:
#   1. (sudo) turns Bluetooth audio off for the lightdm user only and restarts its WirePlumber
#   2. turns it on for arduino regardless of the seat (headless), restarts WirePlumber
#   3. reconnects the speaker, makes it the default output, and plays a test sound on it
# Both settings are config files, so they survive reboots. The desktop and the board's
# built-in audio are unchanged.
set -euo pipefail
BOARD="${BOARD:-arduino@airfour.local}"

ssh "$BOARD" 'cat > /tmp/lightdm_no_bluez.sh' <<'EOF'
set -e
DIR=/var/lib/lightdm/.config/wireplumber/wireplumber.conf.d
install -d -o lightdm -g lightdm /var/lib/lightdm/.config /var/lib/lightdm/.config/wireplumber "$DIR"
cat > "$DIR/80-no-bluez.conf" <<'CONF'
# ME193 lock: leave Bluetooth audio to the arduino user (the login screen needs none)
wireplumber.profiles = { main = { monitor.bluez = disabled } }
CONF
chown lightdm:lightdm "$DIR/80-no-bluez.conf"
U=$(id -u lightdm)
runuser -u lightdm -- env XDG_RUNTIME_DIR=/run/user/$U DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$U/bus \
  systemctl --user restart wireplumber || pkill -u lightdm -x wireplumber || true
echo "lightdm WirePlumber restarted without Bluetooth"
EOF
echo "[1/3] lightdm: Bluetooth audio off (enter the board password for sudo)"
ssh -t "$BOARD" 'sudo bash /tmp/lightdm_no_bluez.sh; rm -f /tmp/lightdm_no_bluez.sh'

ssh "$BOARD" bash -s <<'EOF'
set -euo pipefail
echo "[2/3] arduino: Bluetooth audio on"
mkdir -p ~/.config/wireplumber/wireplumber.conf.d
cat > ~/.config/wireplumber/wireplumber.conf.d/80-bluez-headless.conf <<'CONF'
# AirFour runs headless: the active seat belongs to the lightdm login screen, so by default
# this user's WirePlumber never turns on Bluetooth audio. Enable it regardless of the seat.
wireplumber.profiles = {
  main = {
    monitor.bluez.seat-monitoring = disabled
  }
}
CONF
sleep 2
systemctl --user restart wireplumber
sleep 4

echo "[3/3] reconnect the speaker, set it as default, test sound"
MAC=$(python3 -c 'import json; print(json.load(open("/home/arduino/ArduinoApps/me193_lock/data/speaker.json"))["mac"])')
NODE="bluez_output.${MAC//:/_}.1"
bluetoothctl disconnect "$MAC" >/dev/null 2>&1 || true
sleep 3
bluetoothctl --timeout 15 connect "$MAC" >/dev/null 2>&1 || true
ID=""
for i in $(seq 1 15); do
  ID=$(pw-dump 2>/dev/null | python3 -c '
import json, sys
for o in json.load(sys.stdin):
    p = o.get("info", {}).get("props", {})
    if p.get("node.name") == sys.argv[1]:
        print(o["id"]); break' "$NODE" || true)
  [[ -n "$ID" ]] && break
  sleep 1
done
if [[ -z "$ID" ]]; then
  echo "the speaker is still not an audio output. WirePlumber log:"
  journalctl --user -u wireplumber --no-pager --since "-2min" | grep -Ei "bluez|error" | tail -10
  exit 1
fi
wpctl set-default "$ID"
wpctl set-volume "$ID" 1.0
wpctl status | sed -n '/^Audio/,/Sources:/p'
echo "playing 'Access granted' on $NODE"
pw-play --target "$NODE" ~/ArduinoApps/me193_lock/python/sounds/granted.wav
echo "done"
EOF
