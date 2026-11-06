# ME193 midterm: finger-count camera lock

A safe that opens when the camera sees the right finger count, running on an
Arduino UNO Q (board name: AirFour).

Current state: hardware smoke test only (button, servo, onboard LEDs). There is
no camera or model yet.

| Folder | Contents |
| --- | --- |
| `board/` | App Lab app `me193_lock` (`app.yaml`, `sketch/`, `python/`) and `deploy_board.sh` |
| `capture/` | Data capture scripts |
| `data/` | `raw/` and `processed/` (not committed) |
| `training/` | Model training |
| `models/` | Trained models |
| `part1_v_fit/` | Part 1 V-data fit |
| `docs/` | Report notes and figures |

## Wiring

| Part | Connection |
| --- | --- |
| Micro servo | orange signal to D9, red to 5V, brown to GND |
| Arcade button microswitch | NO to D2, COM to GND (`INPUT_PULLUP`, pressed = LOW) |
| Status lights | onboard RGB LEDs LED3 and LED4 (active low) |

## Deploy

```bash
board/deploy_board.sh --sketch                      # school Wi-Fi ("airfour" SSH alias)
BOARD=arduino@192.168.1.173 board/deploy_board.sh --sketch   # home
```

SSH needs the Mac and the board on the same network. Over the USB-C cable, adb
works without Wi-Fi; it comes with the Arduino IDE:
`~/Library/Arduino15/packages/arduino/tools/adb/32.0.0/adb shell`

## Logs

```bash
ssh airfour 'arduino-app-cli app logs ~/ArduinoApps/me193_lock --tail 30'
```

While USB is connected, the sketch's Monitor output goes out the USB port
(`/dev/cu.usbmodem*` on the Mac), not to an SSH `arduino-app-cli monitor`.
