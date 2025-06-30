import cv2
import numpy as np
import argparse
import os
import time

OUTPUT_FILENAME = "output.txt"
WARP_SIZE = 600
GRID_ROWS = GRID_COLS = 3
NUM_ROBOTS = GRID_ROWS * GRID_COLS
ANALYSIS_DURATION_SEC = 60
START_STABILIZATION_THRESHOLD = 0.01
START_MIN_STABLE_FRAMES = 10
START_CHECK_DURATION_SEC = 5
MOTION_BG_SUB_HISTORY = 60
MOTION_BG_SUB_THRESHOLD = 18
MOTION_AREA_THRESHOLD_PERCENT = 0.63
ANGLE_THRESH_DEG = 4.2
DIFF_PIXEL_THRESH = 10
DIFF_AREA_PERCENT = 0.38
MOTION_SECOND_THRESHOLD_PERCENT = 20
WARMUP_SECONDS = 2
START_DELAY_SEC = 1
RESET_PERIOD_SEC = 20
SHOW_DEBUG = True
SHOW_HOMOGRAPHY_DEBUG = True
LANDING_HIGH_DIFF_THRESH = 0.08
LANDING_LOW_DIFF_THRESH = 0.01
LANDING_STABLE_FRAMES = 10
HARRIS_BLOCK_SIZE = 2
HARRIS_KSIZE = 3
HARRIS_K = 0.04
HARRIS_THRESHOLD_RATIO = 0.1
MIN_HARRIS_CORNERS = 50

def order_pts(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered

def compute_homography_harris(frame):
    print("Computing homography using Harris corners...")
    start_time = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)
    harris_response = cv2.cornerHarris(gray_float, HARRIS_BLOCK_SIZE, HARRIS_KSIZE, HARRIS_K)
    harris_vis_norm = None
    if SHOW_HOMOGRAPHY_DEBUG:
        harris_vis_norm = cv2.normalize(harris_response, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    threshold = HARRIS_THRESHOLD_RATIO * harris_response.max()
    locations = np.where(harris_response > threshold)
    if len(locations[0]) < MIN_HARRIS_CORNERS:
        print(f"Warning: Found only {len(locations[0])} Harris corners (min required: {MIN_HARRIS_CORNERS}). Trying Canny fallback...")
        return None
    corner_points = np.array(list(zip(locations[1], locations[0])), dtype=np.float32)
    print(f"Found {len(corner_points)} Harris corners above threshold.")
    harris_vis_points = frame.copy() if SHOW_HOMOGRAPHY_DEBUG else None
    if SHOW_HOMOGRAPHY_DEBUG:
        for x, y in corner_points:
            cv2.circle(harris_vis_points, (int(x), int(y)), 3, (0, 0, 255), -1)
    try:
        rect = cv2.minAreaRect(corner_points)
    except Exception as e:
        print(f"Error during cv2.minAreaRect on Harris corners: {e}")
        return None
    src_pts = cv2.boxPoints(rect)
    src_ordered = order_pts(src_pts)
    dst_pts = np.array([
        [0, 0],
        [WARP_SIZE - 1, 0],
        [WARP_SIZE - 1, WARP_SIZE - 1],
        [0, WARP_SIZE - 1]
    ], dtype="float32")
    H = cv2.getPerspectiveTransform(src_ordered, dst_pts)
    end_time = time.time()
    print(f"Homography computed successfully using Harris corners in {end_time - start_time:.4f} seconds.")
    if SHOW_HOMOGRAPHY_DEBUG:
         warped_debug = cv2.warpPerspective(frame, H, (WARP_SIZE, WARP_SIZE))
         cv2.rectangle(harris_vis_points, (int(rect[0][0]-rect[1][0]/2), int(rect[0][1]-rect[1][1]/2)), \
                      (int(rect[0][0]+rect[1][0]/2), int(rect[0][1]+rect[1][1]/2)), (0,255,0), 2)
         for i, pt in enumerate(src_ordered):
              cv2.putText(harris_vis_points, f"{i}", (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    return H

def build_rois():
    cell, m = WARP_SIZE // GRID_ROWS, 5
    idx_map = [
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 0), (1, 0), (2, 0)
    ]
    rois = []
    label_map = {}
    current_label = 1
    for r, c in idx_map:
        x = c * cell + m
        y = r * cell + m
        roi_coords = (x, y, cell - 2 * m, cell - 2 * m)
        rois.append(roi_coords)
        roi_index = len(rois) - 1
        label_map[roi_index] = current_label
        current_label += 1
    return rois, label_map

def roi_orientation(gray):
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5: return None
    try:
        ellipse = cv2.fitEllipse(c)
        return ellipse[-1]
    except cv2.error:
        return None

def write_output(results_matrix, filename):
    print(f"Writing results to {filename}...")
    num_seconds = len(results_matrix)
    num_robots = len(results_matrix[0]) if num_seconds > 0 else 0
    if num_robots != NUM_ROBOTS:
         print(f"Error: Results matrix has incorrect number of robots ({num_robots}). Cannot write output.")
         return
    if num_seconds != ANALYSIS_DURATION_SEC:
         print(f"Warning: Results matrix has {num_seconds} seconds, expected {ANALYSIS_DURATION_SEC}.")
         if num_seconds < ANALYSIS_DURATION_SEC:
             results_matrix.extend([[0]*NUM_ROBOTS for _ in range(ANALYSIS_DURATION_SEC - num_seconds)])
         else:
             results_matrix = results_matrix[:ANALYSIS_DURATION_SEC]
         num_seconds = ANALYSIS_DURATION_SEC
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            header = "Saniye\t" + " ".join([f"Robot-{i+1}" for i in range(NUM_ROBOTS)])
            f.write(header + "\n")
            for sec in range(num_seconds):
                if sec < 9:
                    sec_str = f"{sec+1})"
                elif sec < 100:
                    x = "  "
                    sec_str = x + f"{sec+1})"
                padding = 4 - len(sec_str)
                line_start = " " * padding + sec_str
                robot_vals = " ".join(["\t"+f"{results_matrix[sec][r]:>4}" for r in range(NUM_ROBOTS)])
                f.write(line_start + robot_vals + "\n")
        print("Successfully wrote results.")
    except IOError as e:
        print(f"Error writing output file {filename}: {e}")

def detect_landing_frame(cap, fps,
                         high_thr=LANDING_HIGH_DIFF_THRESH,
                         low_thr=LANDING_LOW_DIFF_THRESH,
                         stable=LANDING_STABLE_FRAMES,
                         max_scan_sec=8):
    print(f"Detecting landing frame (max {max_scan_sec}s)...")
    max_f = int(max_scan_sec * fps)
    prev_g = None
    stable_cnt = 0
    found_frame = 0
    state = "INITIAL_SCAN"
    for i in range(max_f):
        ok, fr = cap.read()
        if not ok:
            print("Warning: Video ended during landing detection.")
            break
        g = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if prev_g is not None:
            diff = cv2.absdiff(g, prev_g)
            diff_ratio = np.count_nonzero(diff > 15) / diff.size
            if state == "INITIAL_SCAN":
                if diff_ratio > high_thr:
                    state = "SAW_HIGH_DIFF"
                    stable_cnt = 0
            elif state == "SAW_HIGH_DIFF":
                if diff_ratio < low_thr:
                    state = "STABILIZING"
                    stable_cnt = 1
                elif diff_ratio > high_thr:
                     stable_cnt = 0
            elif state == "STABILIZING":
                if diff_ratio < low_thr:
                    stable_cnt += 1
                    if stable_cnt >= stable:
                        found_frame = i - stable + 1
                        print(f"Landing detected. Stabilization started at frame: {found_frame}")
                        break
                elif diff_ratio > high_thr:
                    state = "SAW_HIGH_DIFF"
                    stable_cnt = 0
                else:
                    state = "SAW_HIGH_DIFF"
                    stable_cnt = 0
        prev_g = g
    if found_frame == 0:
         print("Warning: Stable landing sequence not clearly detected within scan duration. Starting from frame 0.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return found_frame

def process_video(path, WAIT_MS=1):
    global SHOW_DEBUG, SHOW_HOMOGRAPHY_DEBUG
    show_debug = SHOW_DEBUG
    if not os.path.exists(path):
        print(f"Error: Video file not found at {path}"); return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {path}"); return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Warning: Could not get FPS from video. Assuming 30 FPS.")
        fps = 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Info: {frame_width}x{frame_height}, {fps:.2f} FPS")
    start_f = detect_landing_frame(cap, fps)
    print(f"Analysis target start frame (after landing): {start_f}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    ok, first_frame = cap.read()
    if not ok:
        print(f"Error: Could not read frame at the detected start position ({start_f}).")
        cap.release()
        return
    H = compute_homography_harris(first_frame)
    if H is None:
        print("Error: Homography computation failed. Exiting.")
        cap.release()
        if SHOW_HOMOGRAPHY_DEBUG: cv2.destroyAllWindows()
        return
    if SHOW_HOMOGRAPHY_DEBUG:
        print("Homography debug windows shown. Press any key in a debug window to continue analysis...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        SHOW_HOMOGRAPHY_DEBUG = False
    rois, label_map = build_rois()
    analysis_start_frame = start_f + int(fps * START_DELAY_SEC)
    print(f"BG Subtractor warmup starts at frame: {analysis_start_frame}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, analysis_start_frame)
    bgs = [cv2.createBackgroundSubtractorMOG2(
               history=MOTION_BG_SUB_HISTORY,
               varThreshold=MOTION_BG_SUB_THRESHOLD,
               detectShadows=False) for _ in range(NUM_ROBOTS)]
    prev_ang = [None] * NUM_ROBOTS
    prev_warp_gray = None
    print(f"Warming up BG subtractors for {WARMUP_SECONDS} seconds...")
    warmup_frames = int(fps * WARMUP_SECONDS)
    for i in range(warmup_frames):
        ok, fr = cap.read()
        if not ok:
             print("Warning: Video ended during BG warmup.")
             break
        try:
            warp = cv2.warpPerspective(fr, H, (WARP_SIZE, WARP_SIZE))
            if i == 0:
                 prev_warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
            for roi_idx, (x, y, w, h) in enumerate(rois):
                 roi_patch = warp[y:y+h, x:x+w]
                 if roi_patch.size > 0:
                     _ = bgs[roi_idx].apply(roi_patch)
        except cv2.error as e:
             print(f"Error during BG warmup warp/apply (frame {i}): {e}")
             continue
    if ok:
         try:
              warp = cv2.warpPerspective(fr, H, (WARP_SIZE, WARP_SIZE))
              prev_warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
         except cv2.error as e:
              print(f"Error setting last prev_warp_gray after warmup: {e}")
    current_second = 0
    frame_in_second_count = 0
    motion_buffer = [[False] * NUM_ROBOTS for _ in range(int(fps) + 1)]
    results_per_second = []
    print(f"Starting {ANALYSIS_DURATION_SEC} seconds analysis...")
    analysis_frame_index = 0
    while current_second < ANALYSIS_DURATION_SEC:
        ok, fr = cap.read()
        if not ok:
            print(f"Warning: Video ended during analysis at second {current_second+1}.")
            break
        try:
            warp = cv2.warpPerspective(fr, H, (WARP_SIZE, WARP_SIZE))
            warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
            vis = warp.copy() if show_debug else None
        except cv2.error as e:
            print(f"Error during warpPerspective in main loop (frame {analysis_start_frame + analysis_frame_index}): {e}")
            analysis_frame_index += 1
            frame_in_second_count += 1
            if frame_in_second_count >= fps:
                 pass
            else:
                 continue
        current_frame_motion = [False] * NUM_ROBOTS
        for roi_idx, (x, y, w, h) in enumerate(rois):
            roi = warp[y:y+h, x:x+w]
            gray_roi = warp_gray[y:y+h, x:x+w]
            if roi.size == 0 or gray_roi.size == 0:
                 continue
            moved_fg = False
            try:
                fg_mask = bgs[roi_idx].apply(roi)
                motion_area_pixels = cv2.countNonZero(fg_mask)
                roi_area = w * h
                if roi_area > 0:
                    motion_percent = (motion_area_pixels / roi_area) * 100
                    moved_fg = motion_percent > MOTION_AREA_THRESHOLD_PERCENT
            except cv2.error as e:
                print(f"Error in BG subtractor for ROI {roi_idx+1}: {e}")
            moved_ang = False
            current_angle = roi_orientation(gray_roi)
            if current_angle is not None and prev_ang[roi_idx] is not None:
                delta_angle = abs(current_angle - prev_ang[roi_idx])
                angle_diff = min(delta_angle, 360 - delta_angle)
                angle_diff_shortest = min(angle_diff, 180 - (angle_diff if angle_diff>180 else 360-angle_diff))
                delta = abs(current_angle - prev_ang[roi_idx])
                angle_diff = min(delta, 360 - delta)
                if angle_diff >= ANGLE_THRESH_DEG:
                    moved_ang = True
            prev_ang[roi_idx] = current_angle
            moved_diff = False
            if prev_warp_gray is not None:
                try:
                    prev_gray_roi = prev_warp_gray[y:y+h, x:x+w]
                    if gray_roi.shape == prev_gray_roi.shape:
                        diff_frame = cv2.absdiff(gray_roi, prev_gray_roi)
                        diff_pixels = np.count_nonzero(diff_frame > DIFF_PIXEL_THRESH)
                        roi_area = w * h
                        if roi_area > 0:
                            diff_percent = (diff_pixels / roi_area) * 100
                            moved_diff = diff_percent > DIFF_AREA_PERCENT
                except IndexError:
                     print(f"IndexError getting previous ROI {roi_idx+1} for diff.")
                except cv2.error as e:
                     print(f"cv2 error during frame diff ROI {roi_idx+1}: {e}")
            votes = int(moved_fg) + int(moved_ang) + int(moved_diff)
            moving_now = votes >= 2
            current_frame_motion[roi_idx] = moving_now
            if show_debug and vis is not None:
                robot_label = label_map[roi_idx]
                col = (0, 255, 0) if moving_now else (0, 0, 255)
                cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
                cv2.putText(vis, str(robot_label), (x + 5, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        buffer_index = frame_in_second_count % len(motion_buffer)
        motion_buffer[buffer_index] = current_frame_motion
        prev_warp_gray = warp_gray.copy()
        if analysis_frame_index > 0 and analysis_frame_index % int(RESET_PERIOD_SEC * fps) == 0:
            print(f"Resetting background subtractors at analysis frame {analysis_frame_index}...")
            for roi_idx, (x, y, w, h) in enumerate(rois):
                 roi = warp[y:y+h, x:x+w]
                 if roi.size > 0:
                     try:
                         bgs[roi_idx].apply(roi, learningRate=0)
                     except cv2.error as e:
                          print(f"Error during BG reset apply for ROI {roi_idx+1}: {e}")
        if show_debug and vis is not None:
            cv2.putText(vis, f"Second: {current_second+1}/{ANALYSIS_DURATION_SEC}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(vis, f"Frame in Sec: {frame_in_second_count+1}/{int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("ROIs & Detections (q = gizle/hızlı)", vis)
            key = cv2.waitKey(WAIT_MS) & 0xFF
            if key == ord('q'):
                show_debug = False
                cv2.destroyAllWindows()
            elif key == ord(' '):
                 print("Paused. Press space again to continue.")
                 while True:
                      if cv2.waitKey(0) & 0xFF == ord(' '): break
        frame_in_second_count += 1
        analysis_frame_index += 1
        process_final_second = (not ok and current_second < ANALYSIS_DURATION_SEC and len(results_per_second) == current_second)
        if frame_in_second_count >= fps or process_final_second:
            if current_second < ANALYSIS_DURATION_SEC:
                num_frames_in_buffer = frame_in_second_count if not process_final_second else frame_in_second_count % len(motion_buffer)
                if num_frames_in_buffer == 0: num_frames_in_buffer = len(motion_buffer)
                second_result = [0] * NUM_ROBOTS
                threshold_percent = MOTION_SECOND_THRESHOLD_PERCENT
                for r in range(NUM_ROBOTS):
                    motion_count = sum(motion_buffer[f][r] for f in range(num_frames_in_buffer))
                    motion_percentage = (motion_count / num_frames_in_buffer) * 100 if num_frames_in_buffer > 0 else 0
                    if motion_percentage >= threshold_percent:
                        second_result[r] = 1
                results_per_second.append(second_result)
                print(f"Second {current_second+1}/{ANALYSIS_DURATION_SEC} -> {second_result}")
                frame_in_second_count = 0
                current_second += 1
            else:
                break
    cap.release()
    if show_debug:
        cv2.destroyAllWindows()
    print(f"Analysis complete. Total seconds processed: {len(results_per_second)}")
    write_output(results_per_second, OUTPUT_FILENAME)

if __name__ == "__main__":
    # BURAYA VİDEO DOSYASI YOLUNU GİREBİLİRSİNİZ
    video_file = './video.mp4'
    # BURAYA BEKLEME SÜRESİ GİRİLEBİLİR STANDARTI 1MS 120 240 VS GİRİLİP YAVAŞLATILABİLİR
    WAIT_MS = 1
    try:
        video_file = video_file 
    except NameError:
        video_file = None
    ap = argparse.ArgumentParser(description="TUSAS Homework-2 Motion Detector")
    ap.add_argument("video_file", nargs="?", help="Path to the input video file")
    ap.add_argument("--wait-ms", type=int, default=1, help="cv2.waitKey için bekleme süresi (ms) [varsayılan: 1]")
    args = ap.parse_args()
    if args.video_file is not None:
        video_path = args.video_file
    elif video_file is not None:
        video_path = video_file
    else:
        print("Error: Video file must be specified as an argument or by setting the video_file variable.")
        exit(1)
    if hasattr(args, 'quiet') and args.quiet:
        SHOW_DEBUG = False
        SHOW_HOMOGRAPHY_DEBUG = False
    process_video(video_path,WAIT_MS=WAIT_MS)
    print("Processing finished.")