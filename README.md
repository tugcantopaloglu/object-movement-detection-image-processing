# Robot Motion Analysis

This project detects the motion of robots positioned on a **3 × 3 grid** in a video file.  
Using **Python** and **OpenCV**, it analyses which robots move each second and writes the results as a table to a text file.

---

## Features

- Motion detection for **9 robots** on a 3×3 grid
- Homography (perspective correction) with **Harris corner detection**
- Motion detection via background subtraction, angle difference and pixel differencing
- Second‑by‑second results saved in a table
- Visual debug overlay during video playback

---

## Usage

> **Tip:** Open the project folder in _Visual Studio Code → “Open Folder”_.  
> Then the `video_file` variable can easily point to videos placed in the same directory.  
> Otherwise supply the video path via the command line.  
> A sample video is provided as **`video.mp4`** in the root.  
> If you rename it, update `video_file` (around line 397).

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run:

   ```bash
   python main.py tusas-odev2-test.mp4
   ```

   or set `video_file` directly inside the code (line 397).

   To slow playback for detailed inspection, increase `WAIT_MS` (line 399, default **1 ms**).

Upon completion the script creates **`tusas-odev2-ogr.txt`** in the project folder.  
You can grade it with `odev_kontrol.py`.

### Arguments

| Arg          | Description                               |
| ------------ | ----------------------------------------- |
| `video_file` | _required_ – path to the video to analyse |
| `--wait-ms`  | frame delay in debug mode (default **1**) |

Example:

```bash
python main.py tusas-odev2-test.mp4 --wait-ms 120
```

---

## Output Format

```
Second  Robot‑1 Robot‑2 ... Robot‑9
 1)       0       1  ...    0
 2)       1       1  ...    0
 ...
```

Saved to **`tusas-odev2-ogr.txt`**.

---

## Requirements

- Python 3.7+
- OpenCV
- NumPy

---

## Notes

- Provide the correct video name/path; relative paths may fail if the folder is not opened in VS Code.
- The result file is saved in the current working directory.
- Debug windows let you watch the analysis live.
