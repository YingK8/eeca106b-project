#!/usr/bin/env python3
"""
Generate an ArUco marker PDF page for printing.

Examples:
  python3 generate_aruco_marker.py
  python3 generate_aruco_marker.py --marker-id 0 --dict DICT_4X4_50 --size-m 0.05 --dpi 600 --output /tmp/aruco_0.pdf
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

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


PAGE_SIZES_MM = {
    "A4": (210.0, 297.0),
    "LETTER": (215.9, 279.4),
}


def _mm_to_points(mm: float) -> float:
    return (mm / 25.4) * 72.0


def _write_single_image_pdf(
    output_path: Path,
    image_gray: np.ndarray,
    page_width_pt: float,
    page_height_pt: float,
    draw_width_pt: float,
    draw_height_pt: float,
) -> None:
    x_pt = (page_width_pt - draw_width_pt) / 2.0
    y_pt = (page_height_pt - draw_height_pt) / 2.0
    if x_pt < 0 or y_pt < 0:
        raise ValueError("Marker is larger than the page. Reduce --size-m or use larger --page-size.")

    compressed = zlib.compress(image_gray.tobytes())
    content = (
        "q\n"
        f"{draw_width_pt:.6f} 0 0 {draw_height_pt:.6f} {x_pt:.6f} {y_pt:.6f} cm\n"
        "/Im0 Do\n"
        "Q\n"
    ).encode("ascii")

    objects = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width_pt:.6f} {page_height_pt:.6f}] "
            "/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
        ).encode("ascii")
    )
    objects.append(
        (
            f"<< /Type /XObject /Subtype /Image /Width {image_gray.shape[1]} /Height {image_gray.shape[0]} "
            "/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode "
            f"/Length {len(compressed)} >>\nstream\n"
        ).encode("ascii")
        + compressed
        + b"\nendstream"
    )
    objects.append(f"<< /Length {len(content)} >>\nstream\n".encode("ascii") + content + b"endstream")

    pdf = bytearray()
    pdf.extend(b"%PDF-1.4\n")

    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("ascii")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pdf)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an ArUco marker PDF page.")
    parser.add_argument("--marker-id", type=int, default=0, help="ArUco marker id (default: 0).")
    parser.add_argument(
        "--dict",
        type=str,
        default="DICT_4X4_50",
        choices=sorted(ARUCO_DICT_MAP.keys()),
        help="ArUco dictionary.",
    )
    parser.add_argument(
        "--size-m",
        type=float,
        default=0.05,
        help="Physical marker size in meters (default: 0.05).",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Raster DPI for marker generation before embedding in PDF.")
    parser.add_argument("--border-bits", type=int, default=1, help="Marker border width in bits.")
    parser.add_argument(
        "--page-size",
        type=str,
        default="LETTER",
        choices=sorted(PAGE_SIZES_MM.keys()),
        help="PDF page size.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aruco_marker_0.pdf"),
        help="Output PDF path.",
    )
    args = parser.parse_args()

    if args.marker_id < 0:
        raise ValueError("--marker-id must be >= 0")
    if args.size_m <= 0:
        raise ValueError("--size-m must be > 0")
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0")
    if args.border_bits < 1:
        raise ValueError("--border-bits must be >= 1")

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[args.dict])
    size_px = int(round((args.size_m / 0.0254) * args.dpi))
    if size_px < 10:
        raise ValueError("Computed marker image is too small. Increase --size-m or --dpi.")
    marker_img = np.zeros((size_px, size_px), dtype=np.uint8)

    if hasattr(cv2.aruco, "generateImageMarker"):
        cv2.aruco.generateImageMarker(dictionary, args.marker_id, size_px, marker_img, args.border_bits)
    else:
        cv2.aruco.drawMarker(dictionary, args.marker_id, size_px, marker_img, args.border_bits)

    page_w_mm, page_h_mm = PAGE_SIZES_MM[args.page_size]
    page_w_pt = _mm_to_points(page_w_mm)
    page_h_pt = _mm_to_points(page_h_mm)
    marker_size_pt = (args.size_m / 0.0254) * 72.0

    _write_single_image_pdf(
        output_path=args.output,
        image_gray=marker_img,
        page_width_pt=page_w_pt,
        page_height_pt=page_h_pt,
        draw_width_pt=marker_size_pt,
        draw_height_pt=marker_size_pt,
    )

    print(f"Saved marker PDF to {args.output}")
    print(
        f"marker_id={args.marker_id}, dict={args.dict}, size_m={args.size_m:.4f}, "
        f"dpi={args.dpi}, size_px={size_px}, page={args.page_size}, border_bits={args.border_bits}"
    )
    print("Print at 100% scale (disable fit-to-page) and verify marker size with a ruler/caliper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
