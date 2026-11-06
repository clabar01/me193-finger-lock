"""Tests for preprocessing and the capture tools, using a fake camera and scripted keys."""
import json
import re
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "capture")]

import calibrate_roi  # noqa: E402
import capture  # noqa: E402
import cv2  # noqa: E402
import list_cameras  # noqa: E402
import make_grid  # noqa: E402
import numpy as np  # noqa: E402
from common import camera as camera_module  # noqa: E402
from common.preprocess import crop_and_resize, load_roi, preprocess, roi_to_pixels  # noqa: E402

ROI = {"x": 0.25, "y": 0.125, "w": 0.5, "h": 2 / 3, "frame_width": 640, "frame_height": 480}


def make_frame():
    """A noisy frame with a red square from (160, 60) to (480, 380), the ROI's position."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) // 4 + 120
    frame[60:380, 160:480] = (30, 30, 210)
    return frame


class FakeCap:
    def __init__(self):
        self.read_count = 0

    def read(self):
        self.read_count += 1
        return True, make_frame()

    def release(self):
        pass


def scripted_keys(keys, key_times, step_s=0.03):
    """A cv2.waitKey that plays (key, repeats) pairs, spaced like real frames, then quits.

    key_times records when each key was first delivered.
    """
    sequence = []
    for key, repeats in keys:
        sequence += [ord(key) if key else 255] * repeats
    sequence.append(ord("q"))
    iterator = iter(sequence)

    def wait_key(_delay):
        time.sleep(step_s)
        key = next(iterator)
        key_times.setdefault(key, time.time())
        return key
    return wait_key


class PreprocessTests(unittest.TestCase):
    def test_output_format(self):
        x = preprocess(make_frame(), ROI)
        self.assertEqual(x.shape, (3, 128, 128))
        self.assertEqual(x.dtype, np.float32)
        self.assertGreaterEqual(x.min(), -1)
        self.assertLessEqual(x.max(), 1)

    def test_channels_are_rgb(self):
        x = preprocess(make_frame(), ROI)
        self.assertGreater(x[0].mean(), x[2].mean())  # the red square is brightest in channel 0

    def test_saved_crop_matches_live_frame_exactly(self):
        frame = make_frame()
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/crop.png"
            cv2.imwrite(path, crop_and_resize(frame, ROI))
            np.testing.assert_array_equal(preprocess(cv2.imread(path)), preprocess(frame, ROI))

    def test_roi_scales_with_frame(self):
        self.assertEqual(roi_to_pixels(ROI, 640, 480), (160, 60, 480, 380))
        self.assertEqual(roi_to_pixels(ROI, 1280, 960), (320, 120, 960, 760))

    def test_bad_roi_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "roi.json"
            path.write_text(json.dumps({"x": 0.8, "y": 0, "w": 0.5, "h": 0.5}))
            with self.assertRaises(ValueError):
                load_roi(path)


class CalibrateTests(unittest.TestCase):
    def test_auto_detect_finds_red_square(self):
        self.assertEqual(calibrate_roi.find_red_square(make_frame()), (160, 60, 480, 380))

    def test_auto_detect_ignores_non_square_red(self):
        frame = np.full((480, 640, 3), 128, np.uint8)
        frame[200:240, 50:600] = (30, 30, 210)
        self.assertIsNone(calibrate_roi.find_red_square(frame))

    def test_drag_gives_square_inside_frame(self):
        for start, end in (((100, 100), (300, 180)), ((300, 180), (100, 100)), ((600, 400), (10, 479))):
            x0, y0, x1, y1 = calibrate_roi.drag_to_square(start, end, 640, 480)
            self.assertEqual(x1 - x0, y1 - y0)
            self.assertTrue(0 <= x0 <= x1 <= 640 and 0 <= y0 <= y1 <= 480)

    def test_roi_round_trip(self):
        box = (123, 45, 417, 339)
        self.assertEqual(roi_to_pixels(calibrate_roi.box_to_roi(box, 640, 480), 640, 480), box)


class CameraSelectionTests(unittest.TestCase):
    DEVICES = [
        {"index": 0, "name": "icspring camera", "unique_id": "usb-id", "is_usb": True},
        {"index": 1, "name": "FaceTime HD Camera", "unique_id": "built-in", "is_usb": False},
    ]

    def test_finds_camera_by_id_after_index_shift(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "camera.json"
            path.write_text(json.dumps({"index": 0, "name": "icspring camera", "unique_id": "usb-id"}))
            shifted = [{**self.DEVICES[1], "index": 0}, {**self.DEVICES[0], "index": 1}]
            with mock.patch.object(camera_module, "list_mac_devices", return_value=shifted), \
                    mock.patch.object(camera_module, "open_index", side_effect=lambda i: i), \
                    mock.patch.object(camera_module.sys, "platform", "darwin"):
                self.assertEqual(camera_module.open_configured_camera(path)[0], 1)

    def test_missing_camera_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "camera.json"
            path.write_text(json.dumps({"index": 0, "name": "icspring camera", "unique_id": "usb-id"}))
            with mock.patch.object(camera_module, "list_mac_devices", return_value=self.DEVICES[1:]), \
                    mock.patch.object(camera_module.sys, "platform", "darwin"), \
                    self.assertRaises(camera_module.CameraError):
                camera_module.open_configured_camera(path)

    def test_y_cannot_choose_non_usb_camera(self):
        with mock.patch.object(list_cameras, "open_index", return_value=FakeCap()), \
                mock.patch.object(list_cameras.cv2, "imshow"), \
                mock.patch.object(list_cameras.cv2, "waitKey", side_effect=[ord("y"), ord("n")]):
            self.assertEqual(list_cameras.preview(self.DEVICES[1]), "next")

    def test_y_chooses_usb_camera(self):
        with mock.patch.object(list_cameras, "open_index", return_value=FakeCap()), \
                mock.patch.object(list_cameras.cv2, "imshow"), \
                mock.patch.object(list_cameras.cv2, "waitKey", side_effect=[ord("y")]):
            self.assertEqual(list_cameras.preview(self.DEVICES[0]), "yes")


class CaptureLoopTests(unittest.TestCase):
    def run_capture(self, keys, target=200, person="te_st/../x", step_s=0.03):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        settings = {"exposure": "locked", "white_balance": "AUTO", "mean_brightness_at_lock": 100}
        shown = []
        self.key_times = {}
        with mock.patch.object(capture, "open_configured_camera", return_value=(FakeCap(), {"name": "fake"})), \
                mock.patch.object(capture, "lock_and_report", return_value=settings), \
                mock.patch.object(capture, "restore_auto"), \
                mock.patch.object(capture, "ROI_PATH", Path(__file__)), \
                mock.patch.object(capture, "load_roi", return_value=ROI), \
                mock.patch.object(capture.cv2, "imshow", side_effect=lambda w, img: shown.append(img)), \
                mock.patch.object(capture.cv2, "waitKey", scripted_keys(keys, self.key_times, step_s)), \
                mock.patch.object(capture.cv2, "destroyAllWindows"):
            code = capture.main(["--person", person, "--hand", "r", "--session", "desk lamp_1",
                                 "--target", str(target), "--data-root", str(root)])
        self.assertEqual(code, 0)
        self.shown = shown
        return next(p for p in root.iterdir())

    def files(self, folder, label):
        return sorted((folder / label).glob("*.png"))

    def saved_times(self, files):
        """Save times in seconds, read from the timestamps in the filenames."""
        times = []
        for path in files:
            stamp = re.search(r"(\d{8}-\d{6}-\d{3})\.png$", path.name).group(1)
            times.append(datetime.strptime(stamp + "000", "%Y%m%d-%H%M%S-%f").timestamp())
        return times

    def assert_about_five_per_second(self, files):
        self.assertGreaterEqual(len(files), 5)
        gaps = np.diff(self.saved_times(files))
        self.assertGreaterEqual(gaps.mean(), 0.19)
        self.assertLessEqual(gaps.mean(), 0.27)

    def test_tap_saves_one_correct_crop(self):
        folder = self.run_capture([("3", 1)])
        files = self.files(folder, "3")
        self.assertEqual(len(files), 1)
        np.testing.assert_array_equal(cv2.imread(str(files[0])), crop_and_resize(make_frame(), ROI))
        self.assertEqual(sum(len(self.files(folder, l)) for l in "01245"), 0)

    def test_filename_is_clean_and_parseable(self):
        folder = self.run_capture([("1", 1)])
        self.assertEqual(folder.parent, Path(self.tmp.name))
        name = self.files(folder, "1")[0].name
        self.assertRegex(name, r"^te-st-x_R_desk-lamp-1_\d{8}-\d{6}-\d{3}\.png$")

    def test_holding_key_saves_about_five_per_second(self):
        folder = self.run_capture([("4", 66)])  # key repeats, one per frame
        self.assert_about_five_per_second(self.files(folder, "4"))

    def test_burst_saves_without_holding_and_space_stops(self):
        folder = self.run_capture([("b", 1), ("5", 1), ("", 66), (" ", 1), ("", 30)])
        files = self.files(folder, "5")
        self.assert_about_five_per_second(files)
        # Nothing may be saved after Space, during the 30 frames that follow it.
        self.assertLess(self.saved_times(files)[-1], self.key_times[ord(" ")])

    def test_burst_stops_at_target(self):
        folder = self.run_capture([("b", 1), ("0", 1), ("", 66)], target=3)
        self.assertEqual(len(self.files(folder, "0")), 3)

    def test_undo_deletes_newest_only(self):
        folder = self.run_capture([("1", 1), ("", 10), ("2", 1), ("", 10), ("u", 1)])
        self.assertEqual(len(self.files(folder, "1")), 1)
        self.assertEqual(len(self.files(folder, "2")), 0)

    def test_undo_never_touches_earlier_runs(self):
        store_root = tempfile.TemporaryDirectory()
        self.addCleanup(store_root.cleanup)
        crop = crop_and_resize(make_frame(), ROI)
        first = capture.CropStore(store_root.name, "p", "R", "s")
        old = first.save("1", crop)
        second = capture.CropStore(store_root.name, "p", "R", "s")
        self.assertEqual(second.counts["1"], 1)
        self.assertIsNone(second.undo())
        self.assertTrue(old.exists())

    def test_settings_logged(self):
        folder = self.run_capture([("1", 1)])
        entry = json.loads((folder / "sessions.jsonl").read_text().splitlines()[0])
        self.assertEqual(entry["camera"]["exposure"], "locked")
        self.assertEqual(entry["hand"], "R")

    def test_bad_hand_rejected(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            capture.parse_args(["--person", "a", "--session", "b", "--hand", "x"])


class GridTests(unittest.TestCase):
    def test_grid_shape_and_counts(self):
        with tempfile.TemporaryDirectory() as d:
            store = capture.CropStore(d, "p", "R", "s")
            crop = crop_and_resize(make_frame(), ROI)
            for label in "012345":
                store.save(label, crop)
            grid, counts, bad = make_grid.build_grid(Path(d) / "p", 2)
            self.assertEqual(counts, dict.fromkeys("012345", 1))
            self.assertEqual(bad, [])
            self.assertEqual(grid.shape[0], 6 * 136)


if __name__ == "__main__":
    unittest.main()
