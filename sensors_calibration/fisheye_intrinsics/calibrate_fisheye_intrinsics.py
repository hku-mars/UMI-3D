#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust fisheye intrinsic calibration from checkerboard images.

Features:
- Detect checkerboard corners from a folder of images
- Perform robust OpenCV fisheye calibration with automatic bad-frame dropping
- Optionally remove worst reprojection-error frames and recalibrate
- Save calibration YAML and reprojection report

Example:
    python3 calibrate_fisheye_intrinsics.py \
        --image_glob "images/*.png" \
        --checkerboard_cols 6 \
        --checkerboard_rows 9 \
        --square_size 0.10 \
        --output_dir calib_output
"""

import os
import glob
import re
import cv2
import argparse
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def pbar(it, **kwargs):
    return tqdm(it, **kwargs) if tqdm is not None else it


# =========================
# OpenCV fisheye calibration flags
# =========================
CALIB_FLAGS_STRICT = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    | cv2.fisheye.CALIB_CHECK_COND
    | cv2.fisheye.CALIB_FIX_SKEW
)

CALIB_FLAGS_LOOSE = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    | cv2.fisheye.CALIB_FIX_SKEW
)

TERM_CRIT = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    200,
    1e-7,
)


@dataclass
class FrameData:
    """One valid checkerboard observation."""
    path: str
    corners: np.ndarray   # (1, N, 2), float64
    objp: np.ndarray      # (1, N, 3), float64
    size: Tuple[int, int] # (W, H)


def setup_logger(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


def make_objp(checkerboard: Tuple[int, int], square_size: float) -> np.ndarray:
    """
    Create checkerboard 3D points on z=0 plane.

    Args:
        checkerboard: (cols_inner, rows_inner)
        square_size: checker size in meters

    Returns:
        objp: shape (1, N, 3)
    """
    cols, rows = checkerboard
    objp = np.zeros((1, cols * rows, 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def detect_corners(
    img_path: str,
    checkerboard: Tuple[int, int],
    objp: np.ndarray
) -> Optional[FrameData]:
    """
    Detect checkerboard corners from one image.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        logging.warning("Failed to read image: %s", img_path)
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "findChessboardCornersSB"):
        ret, corners = cv2.findChessboardCornersSB(
            gray,
            checkerboard,
            flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        )
    else:
        ret, corners = cv2.findChessboardCorners(
            gray,
            checkerboard,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    50,
                    1e-6,
                ),
            )

    if not ret:
        return None

    corners = corners.reshape(1, -1, 2).astype(np.float64)
    h, w = gray.shape[:2]
    return FrameData(path=img_path, corners=corners, objp=objp, size=(w, h))


def frame_spread_score(frame: FrameData) -> float:
    """
    Heuristic score for checkerboard spatial coverage in image.
    Larger is better.
    """
    W, H = frame.size
    pts = frame.corners.reshape(-1, 2)

    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    area = max(1.0, float((xmax - xmin) * (ymax - ymin)))
    area_norm = area / float(W * H)

    cx, cy = W * 0.5, H * 0.5
    mean_r = float(np.mean(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))
    mean_r_norm = mean_r / float(np.hypot(cx, cy))

    return area_norm + 0.05 * mean_r_norm


def calibrate_fisheye(
    frames: List[FrameData],
    flags: int = CALIB_FLAGS_STRICT
):
    """
    Run OpenCV fisheye calibration.
    """
    objpoints = [f.objp for f in frames]
    imgpoints = [f.corners for f in frames]
    img_shape = frames[0].size  # (W, H)

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3)) for _ in range(len(frames))]
    tvecs = [np.zeros((1, 1, 3)) for _ in range(len(frames))]

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        img_shape,
        K,
        D,
        rvecs,
        tvecs,
        flags=flags,
        criteria=TERM_CRIT,
    )
    return rms, K, D, rvecs, tvecs


def calibrate_fisheye_robust(
    frames: List[FrameData],
    max_drop: int = 60,
    min_frames: int = 15,
    strict_flags: int = CALIB_FLAGS_STRICT,
    loose_flags: int = CALIB_FLAGS_LOOSE,
    stage_name: str = "Calibration",
):
    """
    Robust fisheye calibration with automatic frame dropping.

    Strategy:
    - Try strict calibration first
    - If OpenCV reports an input array index, drop that frame
    - Otherwise drop the frame with worst spread score
    - After many failures, fall back to loose flags
    """
    frames = list(frames)
    dropped = []
    flags = strict_flags

    def drop_by_index(idx: int, reason: str):
        nonlocal frames, dropped
        idx = max(0, min(idx, len(frames) - 1))
        path = frames[idx].path
        logging.warning("%s: %s. Dropping: %s", stage_name, reason, path)
        dropped.append(path)
        frames.pop(idx)

    for attempt in range(max_drop + 1):
        if len(frames) < min_frames:
            raise RuntimeError(
                f"{stage_name}: too few frames left ({len(frames)}) "
                f"after dropping {len(dropped)}"
            )

        try:
            rms, K, D, rvecs, tvecs = calibrate_fisheye(frames, flags=flags)
            return rms, K, D, rvecs, tvecs, dropped, frames

        except cv2.error as e:
            msg = str(e)

            m = re.search(r"input array (\d+)", msg)
            if m is not None:
                drop_by_index(int(m.group(1)), "Ill-conditioned input array")
                continue

            scores = [frame_spread_score(f) for f in frames]
            worst_idx = int(np.argmin(scores))
            drop_by_index(worst_idx, "Calibration unstable (fallback heuristic)")

            if attempt >= max_drop // 2 and flags != loose_flags:
                logging.warning(
                    "%s: falling back to loose flags (without CALIB_CHECK_COND).",
                    stage_name,
                )
                flags = loose_flags

    raise RuntimeError(f"{stage_name}: still failing after dropping {max_drop} frames.")


def reproj_error_per_frame(
    frame: FrameData,
    K: np.ndarray,
    D: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute mean / RMS / max reprojection error in pixels for one frame.
    """
    proj, _ = cv2.fisheye.projectPoints(frame.objp, rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)
    obs = frame.corners.reshape(-1, 2)
    err = np.linalg.norm(proj - obs, axis=1)

    mean_e = float(np.mean(err))
    rms_e = float(np.sqrt(np.mean(err ** 2)))
    max_e = float(np.max(err))
    return mean_e, rms_e, max_e


def save_yaml(
    K: np.ndarray,
    D: np.ndarray,
    out_path: str,
    res_wh: Tuple[int, int]
) -> None:
    """
    Save intrinsics in a compact YAML-style text format.
    Compatible with EquidistantCamera-style config.
    """
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    k1, k2, k3, k4 = [float(x) for x in D.reshape(-1)]
    W, H = res_wh

    txt = f"""cam_model: EquidistantCamera
resolution: [{W}, {H}]
intrinsics: [{fx:.10f}, {fy:.10f}, {cx:.10f}, {cy:.10f}]
distortion_coeffs: [{k1:.10f}, {k2:.10f}, {k3:.10f}, {k4:.10f}]
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt)


def save_reprojection_report(
    report_path: str,
    per_frame_stats: List[Tuple[float, float, float, str]]
) -> None:
    """
    Save reprojection errors sorted by worst RMS first.
    """
    with open(report_path, "w", encoding="utf-8") as fw:
        fw.write("rms_px\tmean_px\tmax_px\tpath\n")
        for rms_e, mean_e, max_e, path in per_frame_stats:
            fw.write(f"{rms_e:.4f}\t{mean_e:.4f}\t{max_e:.4f}\t{path}\n")


def collect_frames(
    image_glob: str,
    checkerboard: Tuple[int, int],
    square_size: float
) -> Tuple[List[str], List[FrameData]]:
    """
    Load image paths and detect valid checkerboard frames.
    """
    paths = sorted(glob.glob(image_glob))
    if not paths:
        raise RuntimeError(f"No images found with pattern: {image_glob}")

    objp = make_objp(checkerboard, square_size)
    frames = []

    for p in pbar(paths, desc="Detect corners", unit="img"):
        fd = detect_corners(p, checkerboard, objp)
        if fd is not None:
            frames.append(fd)

    return paths, frames


def run_calibration(args) -> None:
    checkerboard = (args.checkerboard_cols, args.checkerboard_rows)
    os.makedirs(args.output_dir, exist_ok=True)

    paths, frames = collect_frames(
        image_glob=args.image_glob,
        checkerboard=checkerboard,
        square_size=args.square_size,
    )

    logging.info("Detected valid images: %d / %d", len(frames), len(paths))

    if len(frames) < args.min_frames:
        raise RuntimeError(
            f"Too few valid images ({len(frames)}). "
            f"Need at least {args.min_frames} images with successful corner detection."
        )

    # ========= Stage 1 =========
    rms1, K1, D1, rvecs1, tvecs1, dropped1, frames = calibrate_fisheye_robust(
        frames=frames,
        max_drop=args.max_drop,
        min_frames=args.min_frames,
        stage_name="Fisheye Calib #1",
    )

    if dropped1:
        logging.info("Dropped %d unstable frames in Calib #1", len(dropped1))

    logging.info("=== Calib #1 ===")
    logging.info("RMS: %.6f", rms1)
    logging.info("K:\n%s", K1)
    logging.info("D:\n%s", D1.reshape(-1))

    # Reprojection error statistics
    per = []
    for i, f in enumerate(pbar(frames, desc="Compute reproj error", unit="img")):
        mean_e, rms_e, max_e = reproj_error_per_frame(f, K1, D1, rvecs1[i], tvecs1[i])
        per.append((rms_e, mean_e, max_e, f.path))
    per.sort(reverse=True)

    reproj_report_path = os.path.join(args.output_dir, "reproj_report.txt")
    save_reprojection_report(reproj_report_path, per)
    logging.info("Saved reprojection report: %s", reproj_report_path)

    # ========= Stage 2 =========
    drop_n = int(len(frames) * args.drop_ratio)
    keep_n = len(frames) - drop_n

    if drop_n <= 0 or keep_n < args.min_keep:
        logging.info(
            "Skipping Calib #2 (drop_n=%d, keep_n=%d, min_keep=%d)",
            drop_n, keep_n, args.min_keep
        )
        final_K, final_D, final_rms = K1, D1, rms1
        kept = frames
    else:
        worst = set(x[3] for x in per[:drop_n])
        kept = [f for f in frames if f.path not in worst]

        rms2, K2, D2, rvecs2, tvecs2, dropped2, kept = calibrate_fisheye_robust(
            frames=kept,
            max_drop=args.max_drop,
            min_frames=args.min_frames,
            stage_name=f"Fisheye Calib #2 (after dropping worst {drop_n})",
        )

        if dropped2:
            logging.info("Dropped %d unstable frames in Calib #2", len(dropped2))

        logging.info(
            "=== Calib #2 (after dropping worst %.1f%% = %d images) ===",
            args.drop_ratio * 100.0,
            drop_n,
        )
        logging.info("RMS: %.6f", rms2)
        logging.info("K:\n%s", K2)
        logging.info("D:\n%s", D2.reshape(-1))

        final_K, final_D, final_rms = K2, D2, rms2

    # ========= Save outputs =========
    img0 = cv2.imread(kept[0].path, cv2.IMREAD_COLOR)
    if img0 is None:
        raise RuntimeError(f"Failed to read image: {kept[0].path}")

    H, W = img0.shape[:2]

    yaml_path = os.path.join(args.output_dir, "equidistant.yaml")
    save_yaml(final_K, final_D, yaml_path, (W, H))
    logging.info("Saved calibration YAML: %s", yaml_path)

    np.savetxt(os.path.join(args.output_dir, "K.txt"), final_K)
    np.savetxt(os.path.join(args.output_dir, "D.txt"), final_D.reshape(-1, 1))

    with open(os.path.join(args.output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"total_images={len(paths)}\n")
        f.write(f"valid_images={len(frames)}\n")
        f.write(f"final_kept_images={len(kept)}\n")
        f.write(f"final_rms={final_rms:.8f}\n")

    logging.info("Calibration finished successfully.")


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Robust fisheye intrinsic calibration from checkerboard images."
    )

    parser.add_argument(
        "--image_glob",
        type=str,
        default="images/*.png",
        help='Glob pattern for calibration images, e.g. "images/*.png"',
    )
    parser.add_argument(
        "--checkerboard_cols",
        type=int,
        default=6,
        help="Number of checkerboard inner corners along columns.",
    )
    parser.add_argument(
        "--checkerboard_rows",
        type=int,
        default=9,
        help="Number of checkerboard inner corners along rows.",
    )
    parser.add_argument(
        "--square_size",
        type=float,
        default=0.10,
        help="Checker square size in meters.",
    )

    parser.add_argument(
        "--drop_ratio",
        type=float,
        default=0.20,
        help="Ratio of worst reprojection-error frames removed before refinement.",
    )
    parser.add_argument(
        "--min_keep",
        type=int,
        default=40,
        help="Minimum kept frames required for second-stage calibration.",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=15,
        help="Minimum valid frames required during calibration.",
    )
    parser.add_argument(
        "--max_drop",
        type=int,
        default=60,
        help="Maximum number of automatically dropped frames.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="calib_output",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    setup_logger(args.verbose)
    run_calibration(args)


if __name__ == "__main__":
    main()
