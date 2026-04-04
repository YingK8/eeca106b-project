#!/usr/bin/env python3
"""
Continuously estimate and print the transform between an ArUco marker and camera.

Examples:
  python3 stream_aruco_transform.py
  python3 stream_aruco_transform.py --marker-id 0 --marker-size-m 0.05 --print-interval 0.5

Keys:
  q = quit
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
}


def parse_size(text: str) -> Tuple[int, int]:
    if "x" not in text:
        raise argparse.ArgumentTypeError("Use WxH format, e.g. 1920x1080")
    w_str, h_str = text.lower().split("x", 1)
    return int(w_str), int(h_str)


def rotmat_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    trace = np.trace(rot)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat


def configure_v4l2(device_path: str, width: int, height: int, fps: int, fourcc: str) -> None:
    if not shutil.which("v4l2-ctl"):
        print("v4l2-ctl not found; skipping device pre-configuration.")
        return
    cmd = [
        "v4l2-ctl",
        "-d",
        device_path,
        f"--set-fmt-video=width={width},height={height},pixelformat={fourcc}",
        f"--set-parm={fps}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print("v4l2-ctl failed; continuing with OpenCV settings.")
        if result.stderr.strip():
            print(result.stderr.strip())


def load_intrinsics(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Could not open intrinsics file: {path}")
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    fs.release()
    if camera_matrix is None or dist_coeffs is None:
        raise RuntimeError("intrinsics YAML missing camera_matrix/dist_coeffs")
    return camera_matrix.astype(np.float64), dist_coeffs.reshape(-1, 1).astype(np.float64)


def default_intrinsics_path() -> Path:
    return Path(__file__).resolve().parent / "c299_1080p_intrinsics.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuously print ArUco<->camera transform.")
    parser.add_argument("--camera", type=int, default=0, help="VideoCapture index.")
    parser.add_argument("--device", type=str, default=None, help="Video device path (default: /dev/video{camera}).")
    parser.add_argument("--res", type=parse_size, default="1920x1080", help="Resolution WxH.")
    parser.add_argument("--fps", type=int, default=30, help="Target capture FPS.")
    parser.add_argument("--fourcc", type=str, default="MJPG", help="FOURCC format (default MJPG).")
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=default_intrinsics_path(),
        help="Path to camera intrinsics YAML.",
    )
    parser.add_argument("--marker-id", type=int, default=5, help="Marker ID to track.")
    parser.add_argument("--marker-size-m", type=float, default=0.05, help="Marker size in meters.")
    parser.add_argument(
        "--dict",
        type=str,
        default="DICT_5X5_50",
        choices=sorted(ARUCO_DICT_MAP.keys()),
        help="ArUco dictionary.",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        help="Seconds between console prints (default 0.5).",
    )
    parser.add_argument("--axis-length-m", type=float, default=0.03, help="Axis visualization length in meters.")
    parser.add_argument("--no-v4l2-ctl", action="store_true", help="Skip v4l2-ctl pre-configuration.")
    args = parser.parse_args()

    if args.marker_size_m <= 0:
        raise ValueError("--marker-size-m must be > 0")
    if args.print_interval <= 0:
        raise ValueError("--print-interval must be > 0")
    if args.axis_length_m <= 0:
        raise ValueError("--axis-length-m must be > 0")

    width, height = args.res
    device_path = args.device or f"/dev/video{args.camera}"

    camera_matrix, dist_coeffs = load_intrinsics(args.intrinsics)

    if not args.no_v4l2_ctl:
        configure_v4l2(device_path, width, height, args.fps, args.fourcc)

    cap_target = device_path if args.device else args.camera
    cap = cv2.VideoCapture(cap_target, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera target {cap_target}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    print(f"Capture format: {actual_w}x{actual_h} @ {actual_fps:.2f} FPS")
    print(f"Tracking marker id={args.marker_id}, dict={args.dict}, marker_size={args.marker_size_m} m")

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[args.dict])
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        detect_fn = detector.detectMarkers
    else:
        detect_fn = lambda image: cv2.aruco.detectMarkers(image, aruco_dict, parameters=params)  # noqa: E731

    half = args.marker_size_m / 2.0
    marker_points = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float32,
    )

    last_print = 0.0
    window = "ArUco Transform"

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        corners, ids, _ = detect_fn(frame)
        now = time.time()
        printed = False

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            ids_flat = ids.flatten().tolist()
            if args.marker_id in ids_flat:
                idx = ids_flat.index(args.marker_id)
                image_points = corners[idx].reshape(4, 2).astype(np.float32)

                pnp_flag = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE
                ok_pnp, rvec, tvec = cv2.solvePnP(
                    marker_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=pnp_flag,
                )

                if ok_pnp:
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, args.axis_length_m)

                    t_marker_camera = tvec.reshape(3)
                    r_marker_camera, _ = cv2.Rodrigues(rvec)
                    q_marker_camera = rotmat_to_quat_xyzw(r_marker_camera)

                    r_camera_marker = r_marker_camera.T
                    t_camera_marker = -r_camera_marker @ t_marker_camera
                    q_camera_marker = rotmat_to_quat_xyzw(r_camera_marker)

                    overlay = (
                        f"Marker->Cam t=[{t_marker_camera[0]:+.3f}, {t_marker_camera[1]:+.3f}, {t_marker_camera[2]:+.3f}] m"
                    )
                    cv2.putText(frame, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if now - last_print >= args.print_interval:
                        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
                        print(f"[{stamp}] marker->camera")
                        print(
                            "  t (m): "
                            f"[{t_marker_camera[0]:+.6f}, {t_marker_camera[1]:+.6f}, {t_marker_camera[2]:+.6f}]"
                        )
                        print(
                            "  q_xyzw: "
                            f"[{q_marker_camera[0]:+.6f}, {q_marker_camera[1]:+.6f}, {q_marker_camera[2]:+.6f}, {q_marker_camera[3]:+.6f}]"
                        )
                        print(f"[{stamp}] camera->marker")
                        print(
                            "  t (m): "
                            f"[{t_camera_marker[0]:+.6f}, {t_camera_marker[1]:+.6f}, {t_camera_marker[2]:+.6f}]"
                        )
                        print(
                            "  q_xyzw: "
                            f"[{q_camera_marker[0]:+.6f}, {q_camera_marker[1]:+.6f}, {q_camera_marker[2]:+.6f}, {q_camera_marker[3]:+.6f}]"
                        )
                        printed = True

        if (not printed) and (now - last_print >= args.print_interval):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}] marker {args.marker_id} not detected")
            printed = True

        if printed:
            last_print = now

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
