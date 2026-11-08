#!/usr/bin/env bash
# Runs on AirFour as a systemd user service (airfour-speaker.service), outside the lock app.
#
#   - every 60 s, if the paired speaker is not connected, try to connect it
#     (so the JBL comes back after a reboot or after it was switched off and on)
#   - every 4 min while it is connected, play keepalive.wav (a near-silent 40 Hz tone)
#     so the JBL's auto power-off (about 20 min without sound) doesn't trigger
#
# The speaker's address is in ~/ArduinoApps/me193_lock/data/speaker.json (pair_speaker.sh).
# The lock never depends on this: if this service stops, sounds just stop.
APP="$HOME/ArduinoApps/me193_lock"
KEEPALIVE="$APP/python/sounds/keepalive.wav"
RETRY_S=60
KEEPALIVE_S=240
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

mac() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["mac"])' "$APP/data/speaker.json" 2>/dev/null; }

last_keepalive=0
while true; do
  MAC="$(mac)"
  if [[ -n "$MAC" ]]; then
    if bluetoothctl info "$MAC" 2>/dev/null | grep -q "Connected: yes"; then
      now=$(date +%s)
      if (( now - last_keepalive >= KEEPALIVE_S )); then
        node="bluez_output.${MAC//:/_}.1"
        timeout 10 pw-play --target "$node" "$KEEPALIVE" 2>/dev/null && last_keepalive=$now
      fi
    else
      echo "speaker $MAC not connected; trying"
      timeout 20 bluetoothctl connect "$MAC" >/dev/null 2>&1 && echo "speaker connected" && last_keepalive=0
    fi
  fi
  sleep 30
done
