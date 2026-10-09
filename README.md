# Robot Motion Analysis

A Python and OpenCV script for analyzing nine robot positions in a fixed 3 by 3 grid in a recorded video. It combines background subtraction, ellipse orientation changes and frame differencing, then writes a motion flag for each robot for each analysis second. It does not provide a ROS node or a live camera interface.

The repository includes `main.py`, `requirements.txt` and the sample `video.mp4`. There is no grading script or labeled evaluation dataset.

## Setup and usage

Run from the repository directory with a compatible Python installation. Validation used Python 3.13, OpenCV 5.0.0.93 and NumPy 2.5.3. Dependency versions are not pinned; pip selects versions compatible with your Python environment.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS and Linux:

```bash
source .venv/bin/activate
```

Install dependencies and analyze the supplied video without opening windows:

```bash
python -m pip install -r requirements.txt
python main.py video.mp4 --quiet
```

To inspect the debug windows:

```bash
python main.py video.mp4 --wait-ms 120
```

Homography debug windows appear first. Press a key in one of them to start analysis. During playback, press `q` to hide the windows and continue processing, or press space to pause and space again to resume. A graphical desktop and an OpenCV build with GUI support are required for this mode.

| Argument | Behavior |
| --- | --- |
| `video_file` | Optional input path, defaulting to `./video.mp4` |
| `--wait-ms` | Positive debug playback delay in milliseconds, default `1` |
| `--quiet` | Disable all OpenCV debug windows and key waits |

Relative input paths and `output.txt` are resolved against the current working directory. Quote paths containing spaces. Successful processing overwrites `output.txt`; input, calibration, processing or output failures exit unsuccessfully.

## Processing and output

The script scans up to eight seconds for a large frame change followed by stable frames. If it does not find that sequence, it starts from frame zero. Harris corners and their enclosing minimum-area rectangle define a perspective transform to a 600 by 600 image. The script skips one second after the selected start frame, warms up background subtractors for two seconds, then analyzes up to 60 seconds.

For each grid cell, a frame counts as moving when at least two of the three detectors vote for motion. A second receives `1` when at least 20 percent of its processed frames count as moving; otherwise it receives `0`. A final partial second uses only its available frames. Fractional frame rates use cumulative frame counts for second boundaries. Invalid frame-rate metadata falls back to 30 FPS.

Robot labels are arranged as follows when viewed in the corrected image:

```text
7  1  4
8  2  5
9  3  6
```

The output retains the existing `Saniye` column heading and contains nine robot columns and 60 numbered rows. When a short video ends after some analysis frames, remaining rows are padded with zeros. Those padded zeros do not establish that a robot was observed to be stationary. If no analysis frames are available, processing fails without writing a new result file.

## Validation and limits

Run the regression checks with the installed runtime dependencies:

```bash
python -m unittest discover -s tests -v
```

The checks exercise real OpenCV processing on small synthetic images and a lossless FFV1 encoded synthetic video, including motion flags, stationary frames, partial seconds, fractional frame rates and ellipse angle wraparound. The encoded fixture requires FFV1 support in OpenCV's video backend. Additional checks cover capture cleanup, output failures and CLI behavior; GUI calls are mocked in those CLI checks.

This is a heuristic image-processing implementation. The Harris rectangle can include unrelated scene corners, and there is no manual calibration or Canny fallback. It assumes nine visible positions in a stable grid, suitable contrast for contour fitting and a video long enough for warmup and analysis. Compression artifacts and lighting changes may trigger false motion; an MJPEG version of the synthetic fixture produced a false flag in a stationary cell during validation. Thresholds are constants in `main.py` and may need tuning for a different recording. Automated checks and processing of the supplied clip do not establish real-world detection accuracy or validate robot identities against human labels.

The periodic background operation applies the current frame with `learningRate=0`; it does not recreate or clear the background models.

## License

See [LICENSE](LICENSE).
