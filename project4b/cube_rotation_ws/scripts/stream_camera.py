#!/usr/bin/env python3
"""
Stream the camera without any chessboard processing.

Examples:
  python3 stream_camera.py --camera 0 --res 1920x1080 --fps 30 --fourcc MJPG
  python3 stream_camera.py --camera 0 --res 1920x1080 --fps 30 --fourcc MJPG --v4l2-ctl

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


def parse_size(text: str) -> Tuple[int, int]:
    if "x" not in text:
        raise argparse.ArgumentTypeError("Use WxH format, e.g. 1920x1080")
    w_str, h_str = text.lower().split("x", 1)
    return int(w_str), int(h_str)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream camera frames only.")
    parser.add_argument("--camera", type=int, default=0, help="VideoCapture index.")
    parser.add_argument("--res", type=parse_size, default="1920x1080", help="Resolution WxH.")
    parser.add_argument("--fps", type=int, default=30, help="Target capture FPS.")
    parser.add_argument(
        "--fourcc",
        type=str,
        default="MJPG",
        help="FourCC pixel format (default: MJPG).",
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
        "--save",
        type=Path,
        default=None,
        help="Optional path to save a snapshot on startup.",
    )
    args = parser.parse_args()

    width, height = args.res
    device_path = args.device or f"/dev/video{args.camera}"

    if args.v4l2_ctl and shutil.which("v4l2-ctl"):
        fmt = args.fourcc or "MJPG"
        cmd = [
            "v4l2-ctl",
            "-d",
            device_path,
            f"--set-fmt-video=width={width},height={height},pixelformat={fmt}",
            f"--set-parm={args.fps}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0 and result.stderr.strip():
            print("v4l2-ctl failed:", result.stderr.strip())

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("Backend: V4L2")

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(
        f"Capture format: {actual_w}x{actual_h} @ {actual_fps:.2f} FPS, FOURCC={actual_fourcc_str}"
    )

    if args.save:
        ret, frame = cap.read()
        if ret:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save), frame)
            print("Saved snapshot to", args.save)

    fps_last = time.time()
    fps_frames = 0
    fps_value = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        fps_frames += 1
        now = time.time()
        if now - fps_last >= 1.0:
            fps_value = fps_frames / (now - fps_last)
            fps_frames = 0
            fps_last = now

        cv2.putText(
            frame,
            f"FPS: {fps_value:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.imshow("Stream", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
