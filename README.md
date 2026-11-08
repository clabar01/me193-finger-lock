# ME193 midterm: finger-count camera lock

A safe that opens when the camera sees the right finger count, running on an
Arduino UNO Q (board name: AirFour).

The Mac runs the webcam, the CNN, the no-hand check and the lock policy; AirFour
(app `me193_lock`) runs the button, servo, LED matrix and RGB LEDs. They talk over MQTT.

| Folder | Contents |
| --- | --- |
| `board/` | App Lab app `me193_lock` (`app.yaml`, `sketch/`, `python/`) and `deploy_board.sh` |
| `capture/` | Webcam selection, ROI calibration, image capture, contact sheets |
| `common/` | `preprocess.py`, the one preprocessing path, and `camera.py` |
| `config/` | Saved camera choice and ROI |
| `data/` | See [data/README.md](data/README.md); images are not committed |
| `training/` | Model training |
| `models/` | Trained models |
| `part1_v_fit/` | Part 1 V-data fit |
| `docs/` | Report notes and figures |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Capture

Use the external USB webcam only.

```bash
python capture/list_cameras.py     # choose the webcam
python capture/calibrate_roi.py    # mark the red square
python capture/capture.py          # save labeled crops
python capture/make_grid.py --person <name>   # check what was saved
```

## Training

```bash
python training/train.py                  # model, ONNX export, metrics, figures
python training/network_walkthrough.py    # stop-motion frames of one hand
python capture/live_test.py               # live webcam test with the ONNX model
```

In the live test, B takes a snapshot of the empty square (the lock's button will do
this). While the square matches the snapshot (difference below the threshold, default
6, adjust with + and -), the answer is 0 whatever the model says. See `common/no_hand.py`.

Outputs: `models/finger_cnn.pt`, `models/finger_cnn.onnx`, `models/finger_cnn_metrics.json`,
and figures in `docs/figures/`. `data/exclude.txt` lists mislabeled frames that training skips.

## The lock on AirFour alone (no Mac)

```bash
BOARD=arduino@airfour.local board/deploy_board.sh --mode local   # add --sketch if the sketch changed
```

With the webcam on AirFour's USB-C hub, the board runs the camera, the ONNX model, the
no-hand check and the policy itself (`board/python/local_lock.py`) and drives the sketch
through the Bridge; it needs no Mac and no internet. Web page: `http://<board IP>:7000`
(camera, prediction, state, timing, a backup button, **Calibrate ROI**, and a live accuracy
test logged to `logs/live_eval.jsonl` on the board). After moving the camera or the square,
empty the square and press Calibrate ROI. `--mode mac` switches back to the Mac version.
`lock/benchmark.py` times the per-frame work on either machine.

## The lock

```bash
python lock/lock_app.py            # first run creates config/lock.json (code + signing secret)
# set "code" in config/lock.json (3 digits, 1-5), then put the secret on the board:
BOARD=arduino@airfour.local board/deploy_board.sh --sketch
python lock/lock_app.py            # B = button (backup), +/- = no-hand threshold, Q = quit
python lock/fake_board.py          # optional: stand-in for AirFour (Enter = button)
```

Press the arcade button with the square empty (that frame is the no-hand snapshot), then
show the digits with a closed fist between them. An empty square is "no hand": it pauses
reading but does not separate digits. Rules are in
`common/lock_policy.py`; settings (frames, confidence, timeouts) in `config/lock.json`;
servo angles and `DEBUG` at the top of `board/sketch/sketch.ino`. Attempts are logged
to `logs/lock_attempts.jsonl`. `config/lock.json`, the board's `lock_secret.txt` and
`logs/` are not in git: they hold the code, the secret and the digits entered.

MQTT: `broker.hivemq.com:1883`, topics `ME193/ceci/lock/{button,cmd,board}`. The broker is
public, so commands are signed (HMAC-SHA256) and time-stamped; the board rejects forged,
replayed and stale ones (see `board/python/lock_protocol.py`). The code never leaves the Mac.

## Tests

```bash
python -m unittest discover tests
```

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
