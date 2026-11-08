"""Move AirFour's servo to angles you type, to find LOCKED_ANGLE and UNLOCKED_ANGLE for the latch.

    python lock/servo_test.py

Type an angle (0-180) and press Enter; q quits. Run it while lock/lock_app.py
is not running. The command is signed like the lock's (secret in config/lock.json).
The servo stays where you put it until the lock next changes mode.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "lock"), str(ROOT / "board" / "python")]

from lock_app import Link, load_config  # noqa: E402


def main():
    link = Link(load_config()["secret"])
    for _ in range(50):
        if link.connected:
            break
        time.sleep(0.1)
    if not link.connected:
        print("Could not reach the MQTT broker.")
        return 1
    time.sleep(1)  # the board's retained state arrives
    board = link.board
    print(f"AirFour {'online, ' + board.get('state', '?') if board.get('online') else 'OFFLINE'}")
    print("Type an angle 0-180 and press Enter (q quits).")
    try:
        for line in sys.stdin:
            text = line.strip().lower()
            if text in ("q", "quit"):
                break
            try:
                angle = int(text)
            except ValueError:
                print("  an angle is a whole number, 0-180")
                continue
            if not 0 <= angle <= 180:
                print("  out of range: 0-180")
                continue
            link.send("servo", angle)
            print(f"  sent {angle} degrees")
    except KeyboardInterrupt:
        pass
    finally:
        time.sleep(0.5)  # let the last message go out
        link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
