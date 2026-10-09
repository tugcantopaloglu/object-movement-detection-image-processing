import contextlib
import io
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import main


def board_frame(robot_x=300):
    frame = np.full((600, 600, 3), 255, dtype=np.uint8)
    for row in range(3):
        for col in range(3):
            cv2.rectangle(frame, (col * 200 + 5, row * 200 + 5),
                          (col * 200 + 195, row * 200 + 195), (180, 180, 180), -1)
            center = (robot_x, 100) if (row, col) == (0, 1) else (col * 200 + 100, row * 200 + 100)
            cv2.ellipse(frame, center, (20, 35), 0, 0, 360, (0, 0, 0), -1)
    return frame


class FrameCapture:
    def __init__(self, frames, fps=10):
        self.frames = frames
        self.fps = fps
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps,
                cv2.CAP_PROP_FRAME_WIDTH: 600,
                cv2.CAP_PROP_FRAME_HEIGHT: 600}.get(prop, self.position)

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
        return True

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position].copy()
        self.position += 1
        return True, frame

    def release(self):
        self.released = True


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "output.txt"
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(main, "SHOW_DEBUG", False))
        self.stack.enter_context(patch.object(main, "SHOW_HOMOGRAPHY_DEBUG", False))
        self.stack.enter_context(patch.object(main, "OUTPUT_FILENAME", str(self.output)))
        self.log = io.StringIO()
        self.stack.enter_context(contextlib.redirect_stdout(self.log))

    def test_real_video_motion_and_partial_second(self):
        path = Path(self.directory.name) / "motion.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 10, (600, 600))
        self.assertTrue(writer.isOpened())
        try:
            for _ in range(30):
                writer.write(board_frame())
            for index in range(15):
                writer.write(board_frame(240 + index * 8))
        finally:
            writer.release()
        self.assertTrue(main.process_video(str(path)))
        rows = self.output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 61)
        self.assertEqual([int(value) for value in rows[1].split()[1:]], [1] + [0] * 8)
        self.assertEqual([int(value) for value in rows[2].split()[1:]], [1] + [0] * 8)
        self.assertIn("Partial second 2/60", self.log.getvalue())

    def test_static_video_stays_still(self):
        capture = FrameCapture([board_frame()] * 45)
        with patch.object(main.os.path, "exists", return_value=True), patch.object(main.cv2, "VideoCapture", return_value=capture):
            self.assertTrue(main.process_video("synthetic.avi"))
        self.assertTrue(capture.released)
        rows = self.output.read_text(encoding="utf-8").splitlines()
        self.assertTrue(all(int(value) == 0 for row in rows[1:] for value in row.split()[1:]))

    def test_fractional_fps_uses_elapsed_video_time(self):
        capture = FrameCapture([board_frame()] * 17, fps=2.5)
        with patch.object(main.os.path, "exists", return_value=True), patch.object(main.cv2, "VideoCapture", return_value=capture), patch.object(main, "ANALYSIS_DURATION_SEC", 4):
            self.assertTrue(main.process_video("synthetic.avi"))
        self.assertIn("Second 4/4", self.log.getvalue())
        self.assertNotIn("Partial second", self.log.getvalue())

    def test_nonfinite_fps_uses_fallback(self):
        capture = FrameCapture([board_frame()] * 105, fps=float("nan"))
        with patch.object(main.os.path, "exists", return_value=True), patch.object(main.cv2, "VideoCapture", return_value=capture):
            self.assertTrue(main.process_video("synthetic.avi"))
        self.assertIn("Assuming 30 FPS", self.log.getvalue())
        self.assertTrue(capture.released)

    def test_video_without_analysis_frames_does_not_write_results(self):
        capture = FrameCapture([board_frame()] * 30)
        with patch.object(main.os.path, "exists", return_value=True), patch.object(main.cv2, "VideoCapture", return_value=capture):
            self.assertFalse(main.process_video("synthetic.avi"))
        self.assertTrue(capture.released)
        self.assertFalse(self.output.exists())

    def test_harris_accepts_grid_and_rejects_blank_frame(self):
        homography = main.compute_homography_harris(board_frame())
        self.assertEqual(homography.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(homography)))
        self.assertIsNone(main.compute_homography_harris(np.zeros((100, 100, 3), dtype=np.uint8)))

    def test_ellipse_orientation_wraparound(self):
        angles = []
        for angle in (89, 91):
            gray = np.full((200, 200), 255, dtype=np.uint8)
            cv2.ellipse(gray, (100, 100), (55, 25), angle, 0, 360, 0, -1)
            angles.append(main.roi_orientation(gray))
        self.assertTrue(all(angle is not None for angle in angles))
        self.assertGreater(abs(angles[0] - angles[1]), 170)
        self.assertLess(main.orientation_difference(*angles), main.ANGLE_THRESH_DEG)
        self.assertGreater(main.orientation_difference(20, 35), main.ANGLE_THRESH_DEG)

    def test_capture_released_on_processing_exception(self):
        capture = FrameCapture([board_frame()] * 45)
        with patch.object(main.os.path, "exists", return_value=True), patch.object(main.cv2, "VideoCapture", return_value=capture), patch.object(main, "compute_homography_harris", side_effect=RuntimeError("processing failed")):
            with self.assertRaisesRegex(RuntimeError, "processing failed"):
                main.process_video("synthetic.avi")
        self.assertTrue(capture.released)
        self.assertFalse(self.output.exists())

    def test_warp_failure_does_not_reuse_previous_frame(self):
        capture = FrameCapture([board_frame()] * 45)
        warp = main.cv2.warpPerspective
        calls = 0

        def fail_in_analysis(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 21:
                raise cv2.error("analysis warp failed")
            return warp(*args, **kwargs)

        with patch.object(main.os.path, "exists", return_value=True), patch.object(main.cv2, "VideoCapture", return_value=capture), patch.object(main.cv2, "warpPerspective", side_effect=fail_in_analysis):
            self.assertFalse(main.process_video("synthetic.avi"))
        self.assertTrue(capture.released)
        self.assertFalse(self.output.exists())

    def test_output_padding_does_not_mutate_input(self):
        results = [[1] + [0] * 8]
        self.assertTrue(main.write_output(results, self.output))
        self.assertEqual(len(results), 1)
        self.assertEqual(len(self.output.read_text(encoding="utf-8").splitlines()), 61)

    def test_invalid_results_and_unwritable_output_fail(self):
        self.assertFalse(main.write_output([], self.output))
        self.assertFalse(main.write_output([[0] * 9, [0]], self.output))
        self.assertFalse(main.write_output([[0] * 9], Path(self.directory.name)))

    def test_cli_passes_wait_delay(self):
        capture = FrameCapture([board_frame()] * 45)
        with contextlib.chdir(self.directory.name), patch("sys.argv", ["main.py", "input.avi", "--wait-ms", "120"]), patch.object(main.cv2, "VideoCapture", return_value=capture), patch.object(main.os.path, "exists", return_value=True), patch.object(main.cv2, "imshow"), patch.object(main.cv2, "destroyAllWindows"), patch.object(main.cv2, "waitKey", return_value=1) as wait:
            runpy.run_path(str(Path(main.__file__)), run_name="__main__")
        wait.assert_any_call(120)
        self.assertTrue(capture.released)

    def test_cli_quiet_disables_windows(self):
        capture = FrameCapture([board_frame()] * 45)
        with contextlib.chdir(self.directory.name), patch("sys.argv", ["main.py", "input.avi", "--quiet"]), patch.object(main.cv2, "VideoCapture", return_value=capture), patch.object(main.os.path, "exists", return_value=True), patch.object(main.cv2, "imshow") as show, patch.object(main.cv2, "waitKey") as wait:
            runpy.run_path(str(Path(main.__file__)), run_name="__main__")
        show.assert_not_called()
        wait.assert_not_called()
        self.assertTrue(capture.released)

    def test_cli_open_failure_exits_unsuccessfully(self):
        with patch("sys.argv", ["main.py", "input.avi", "--quiet"]), patch.object(main.cv2, "VideoCapture") as capture, patch.object(main.os.path, "exists", return_value=True):
            capture.return_value.isOpened.return_value = False
            with self.assertRaises(SystemExit) as error:
                runpy.run_path(str(Path(main.__file__)), run_name="__main__")
        self.assertEqual(error.exception.code, 1)
        capture.return_value.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
