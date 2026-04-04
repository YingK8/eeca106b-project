#!/usr/bin/env python3
"""
Generate a printable checkerboard image.

Example (10x7 squares, 25 mm squares, A4 at 300 DPI):
  python3 print_checkerboard.py --squares 10x7 --square-mm 25 --dpi 300 \
    --output /tmp/checkerboard_a4.png

Printing tips:
- Print at 100% scale (no "fit to page").
- Measure one square with a caliper after printing.
- Use that measured square size in calibration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def parse_squares(text: str) -> Tuple[int, int]:
    if "x" not in text:
        raise argparse.ArgumentTypeError("Use WxH format, e.g. 10x7")
    w_str, h_str = text.lower().split("x", 1)
    return int(w_str), int(h_str)


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / 25.4))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a checkerboard image for printing.")
    parser.add_argument("--squares", type=parse_squares, default="10x7", help="Squares WxH.")
    parser.add_argument("--square-mm", type=float, default=25.0, help="Square size in mm.")
    parser.add_argument("--dpi", type=int, default=300, help="Dots per inch for output image.")
    parser.add_argument("--margin-mm", type=float, default=10.0, help="White margin around board in mm.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().with_name("calibration_image.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args()

    squares_w, squares_h = args.squares
    square_px = mm_to_px(args.square_mm, args.dpi)
    margin_px = mm_to_px(args.margin_mm, args.dpi)

    board_w = squares_w * square_px
    board_h = squares_h * square_px
    img_w = board_w + 2 * margin_px
    img_h = board_h + 2 * margin_px

    img = np.full((img_h, img_w), 255, dtype=np.uint8)

    for y in range(squares_h):
        for x in range(squares_w):
            if (x + y) % 2 == 0:
                x0 = margin_px + x * square_px
                y0 = margin_px + y * square_px
                img[y0 : y0 + square_px, x0 : x0 + square_px] = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), img)
    print("Saved", args.output)
    print(f"Image size: {img_w}x{img_h} px at {args.dpi} DPI")
    print(f"Square size: {args.square_mm} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
