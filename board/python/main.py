"""me193_lock: hardware smoke test, Python side.

The sketch does all the work (button, servo, onboard LEDs). This side only logs
the status lines it sends with Bridge.notify("lock_status", text), so they show
up in `arduino-app-cli app logs`. The camera and finger-count model go here later.
"""
import time

from arduino.app_utils import App, Bridge, Logger

logger = Logger("me193_lock")


def lock_status(text):
    logger.info(f"sketch: {text}")


def idle_loop():
    time.sleep(1)


logger.info("me193_lock started, waiting for sketch status")
Bridge.provide("lock_status", lock_status)

App.run(user_loop=idle_loop)
