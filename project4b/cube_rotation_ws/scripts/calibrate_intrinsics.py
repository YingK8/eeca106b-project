#!/usr/bin/env python3
"""
Camera intrinsics calibration for a specific resolution.

Usage example (Logitech at 1080p, 9x6 inner corners, 25 mm squares):
  python3 calibrate_intrinsics.py --camera 0 --res 1920x1080 --pattern 9x6 --square-mm 25 \
    --samples 25

Optional v4l2-ctl pre-configuration (opt-in):
  python3 calibrate_intrinsics.py --camera 0 --res 1920x1080 --pattern 9x6 --square-mm 25 \
    --samples 25 --v4l2-ctl

Workflow:
1) Print a checkerboard (see print_checkerboard.py). Measure one square with a caliper.
2) Start this script and set the camera to the exact target resolution.
3) Move the board to many orientations and distances. Avoid only fronto-parallel views.
4) When the board is detected, press 'c' to capture a sample. Aim for 20-30 samples.
5) Press 'q' to calibrate and save.

Keys:
  c = capture sample (only if board found)
  u = undo last sample
  q = finish and calibrate
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def parse_size(text: str) -> Tuple[int, int]:
    if "x" not in text:
        raise argparse.ArgumentTypeError("Use WxH format, e.g. 1920x1080")
    w_str, h_str = text.lower().split("x", 1)
    return int(w_str), int(h_str)


def parse_pattern(text: str) -> Tuple[int, int]:
    if "x" not in text:
        raise argparse.ArgumentTypeError("Use CxR inner-corner format, e.g. 9x6")
    c_str, r_str = text.lower().split("x", 1)
    return int(c_str), int(r_str)


def compute_reprojection_error(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    total_error = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2)
        total_error += err * err
        total_points += len(objpoints[i])
    return float(np.sqrt(total_error / max(total_points, 1)))


def write_opencv_yaml(
    path: Path,
    width: int,
    height: int,
    pattern_cols: int,
    pattern_rows: int,
    square_mm: float,
    rms: float,
    reproj: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> None:
    data_cm = camera_matrix.reshape(-1).tolist()
    data_dc = dist_coeffs.reshape(-1).tolist()
    lines = [
        "%YAML:1.0",
        "---",
        f"image_width: {width}",
        f"image_height: {height}",
        f"pattern_cols: {pattern_cols}",
        f"pattern_rows: {pattern_rows}",
        f"square_mm: {square_mm}",
        f"rms: {rms}",
        f"reprojection_rmse: {reproj}",
        "camera_matrix: !!opencv-matrix",
        "  rows: 3",
        "  cols: 3",
        "  dt: d",
        "  data: [" + ", ".join(f"{v:.12g}" for v in data_cm) + "]",
        "dist_coeffs: !!opencv-matrix",
        f"  rows: 1",
        f"  cols: {len(data_dc)}",
        "  dt: d",
        "  data: [" + ", ".join(f"{v:.12g}" for v in data_dc) + "]",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\\n".join(lines))


class LogitechCam:
    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        fps: int,
        fourcc: str,
        autofocus: bool,
        device_path: str,
        use_v4l2_ctl: bool,
    ) -> None:
        self._lock = threading.Lock()
        self._frame = None
        self._timestamp = 0.0
        self._stop = threading.Event()

        if use_v4l2_ctl and shutil.which("v4l2-ctl"):
            fmt = fourcc or "MJPG"
            cmd = [
                "v4l2-ctl",
                "-d",
                device_path,
                f"--set-fmt-video=width={width},height={height},pixelformat={fmt}",
                f"--set-parm={fps}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                print("v4l2-ctl failed; continuing with OpenCV settings only.")
                if result.stderr.strip():
                    print(result.stderr.strip())
        elif use_v4l2_ctl:
            print("v4l2-ctl not found; continuing with OpenCV settings only.")

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        if not autofocus:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            af_state = self.cap.get(cv2.CAP_PROP_AUTOFOCUS)
            if af_state >= 0:
                print(f"Autofocus set to OFF (reported {af_state}).")
            else:
                print("Autofocus control not supported by this backend/driver.")

        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc_str = "".join(
            [chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)]
        )
        print(
            f"Capture format: {actual_w}x{actual_h} @ {actual_fps:.2f} FPS, FOURCC={actual_fourcc_str}"
        )
        if fourcc and actual_fourcc_str.strip() and actual_fourcc_str != fourcc:
            print(f"Warning: requested FOURCC {fourcc}, got {actual_fourcc_str}.")

        self.actual_width = actual_w
        self.actual_height = actual_h

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            ret = self.cap.grab()
            if not ret:
                time.sleep(0.005)
                continue
            ret, frame = self.cap.retrieve()
            if not ret:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._timestamp = time.time()

    def read_latest(self):
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame, self._timestamp

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.cap.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Camera intrinsics calibration (OpenCV).")
    parser.add_argument("--camera", type=int, default=0, help="VideoCapture index.")
    parser.add_argument("--res", type=parse_size, default="1920x1080", help="Resolution WxH.")
    parser.add_argument("--pattern", type=parse_pattern, default="9x6", help="Inner corners CxR.")
    parser.add_argument("--square-mm", type=float, default=25.0, help="Square size in mm.")
    parser.add_argument("--samples", type=int, default=25, help="Number of samples to collect.")
    parser.add_argument("--fps", type=int, default=30, help="Target capture FPS.")
    parser.add_argument(
        "--fourcc",
        type=str,
        default="MJPG",
        help="FourCC pixel format (default: MJPG for higher FPS at 1080p).",
    )
    parser.add_argument(
        "--detect-scale",
        type=float,
        default=0.5,
        help="Scale factor for chessboard detection (0 < s <= 1).",
    )
    parser.add_argument(
        "--detect-interval",
        type=float,
        default=0.2,
        help="Seconds between chessboard detections (0 for every frame).",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Overlay measured display FPS on the preview window.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Video device path (default: /dev/video{camera}).",
    )
    # By default, do not use v4l2-ctl pre-configuration.
    parser.add_argument(
        "--v4l2-ctl",
        action="store_true",
        help="Enable v4l2-ctl format setup (Linux).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src"
        / "cro_estimation"
        / "assets"
        / "calibration"
        / "c299_1080p_intrinsics_camera13.yaml",
        help="Output YAML path.",
    )
    parser.add_argument(
        "--save-images",
        type=Path,
        default=None,
        help="Optional directory to save captured frames.",
    )
    parser.add_argument(
        "--autofocus",
        action="store_true",
        help="Enable autofocus (default: disabled for calibration).",
    )
    args = parser.parse_args()

    width, height = args.res
    pattern_cols, pattern_rows = args.pattern
    if not (0.0 < args.detect_scale <= 1.0):
        raise ValueError("--detect-scale must be in (0, 1].")

    if args.save_images:
        args.save_images.mkdir(parents=True, exist_ok=True)

    device_path = args.device or f"/dev/video{args.camera}"
    cam = LogitechCam(
        camera_index=args.camera,
        width=width,
        height=height,
        fps=args.fps,
        fourcc=args.fourcc,
        autofocus=args.autofocus,
        device_path=device_path,
        use_v4l2_ctl=args.v4l2_ctl,
    )
    if (cam.actual_width, cam.actual_height) != (width, height):
        print(
            f"Warning: requested {width}x{height}, got {cam.actual_width}x{cam.actual_height}. "
            "Calibration will be for the actual size."
        )
        width, height = cam.actual_width, cam.actual_height

    objp = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    objp *= args.square_mm

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    captured = 0
    last_capture_time = 0.0

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    fps_last_time = time.monotonic()
    fps_frames = 0
    fps_value = 0.0
    last_detect_time = 0.0
    last_found = False
    last_corners_sub = None
    last_frame_ts = 0.0

    while True:
        frame, _ts = cam.read_latest()
        if frame is None:
            time.sleep(0.005)
            continue
        if _ts != last_frame_ts:
            fps_frames += 1
            last_frame_ts = _ts
        now = time.time()
        do_detect = args.detect_interval <= 0 or (now - last_detect_time) >= args.detect_interval
        if do_detect:
            detect_scale = args.detect_scale
            found = False
            corners = None
            gray_full = None

            if detect_scale < 1.0:
                small = cv2.resize(frame, (0, 0), fx=detect_scale, fy=detect_scale)
                gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(
                    gray_small,
                    (pattern_cols, pattern_rows),
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    | cv2.CALIB_CB_NORMALIZE_IMAGE
                    | cv2.CALIB_CB_FAST_CHECK,
                )
                if found:
                    corners = corners / detect_scale
            else:
                gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(
                    gray_full,
                    (pattern_cols, pattern_rows),
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    | cv2.CALIB_CB_NORMALIZE_IMAGE
                    | cv2.CALIB_CB_FAST_CHECK,
                )

            if found:
                if gray_full is None:
                    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                last_corners_sub = cv2.cornerSubPix(
                    gray_full, corners, (11, 11), (-1, -1), criteria
                )
                last_found = True
            else:
                last_found = False
                last_corners_sub = None
            last_detect_time = now

        display = frame.copy()
        recent_detection = (now - last_detect_time) < max(args.detect_interval * 2.0, 0.5)
        if last_found and recent_detection and last_corners_sub is not None:
            cv2.drawChessboardCorners(
                display, (pattern_cols, pattern_rows), last_corners_sub, True
            )

        status = f"Samples: {captured}/{args.samples}"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if args.show_fps:
            now_mono = time.monotonic()
            if now_mono - fps_last_time >= 1.0:
                fps_value = fps_frames / (now_mono - fps_last_time)
                fps_frames = 0
                fps_last_time = now_mono
            cv2.putText(
                display,
                f"FPS: {fps_value:.1f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        cv2.putText(
            display,
            "c=capture  u=undo  q=finish",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("u") and objpoints:
            objpoints.pop()
            imgpoints.pop()
            captured -= 1
            print(f"Removed sample, now {captured} samples.")
        if key == ord("c") and last_found and last_corners_sub is not None and recent_detection:
            if now - last_capture_time < 0.5:
                continue
            objpoints.append(objp.copy())
            imgpoints.append(last_corners_sub)
            captured += 1
            last_capture_time = now
            print(f"Captured {captured}/{args.samples}")
            if args.save_images:
                out_path = args.save_images / f"sample_{captured:02d}.png"
                cv2.imwrite(str(out_path), frame)
        if captured >= args.samples:
            break

    cam.close()
    cv2.destroyAllWindows()

    if captured < 5:
        print("Not enough samples to calibrate. Need at least 5.")
        return 1

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (width, height), None, None
    )
    reproj = compute_reprojection_error(
        objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(args.output), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", width)
    fs.write("image_height", height)
    fs.write("pattern_cols", pattern_cols)
    fs.write("pattern_rows", pattern_rows)
    fs.write("square_mm", args.square_mm)
    fs.write("rms", float(rms))
    fs.write("reprojection_rmse", float(reproj))
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs)
    fs.release()

    if not args.output.exists() or args.output.stat().st_size == 0:
        print("OpenCV FileStorage did not create output; writing fallback YAML.")
        write_opencv_yaml(
            args.output,
            width,
            height,
            pattern_cols,
            pattern_rows,
            args.square_mm,
            float(rms),
            float(reproj),
            camera_matrix,
            dist_coeffs,
        )
    print("Saved calibration to", args.output)
    print("RMS:", rms)
    print("Reprojection RMSE:", reproj)
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coeffs:\n", dist_coeffs.ravel())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
