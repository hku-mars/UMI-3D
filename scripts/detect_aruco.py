#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click
from tqdm import tqdm
import yaml
import json
import av
import numpy as np
import cv2
import pickle
import pandas as pd

from umi.common.cv_util import (
    parse_aruco_config,
    parse_fisheye_intrinsics,
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    draw_predefined_mask
)


def _load_video_timestamps_csv(ts_csv_path: str) -> np.ndarray:
    vdf = pd.read_csv(ts_csv_path)
    if ('frame_idx' not in vdf.columns) or ('t_sec' not in vdf.columns):
        raise RuntimeError(f"timestamps csv must contain columns: frame_idx, t_sec. Got: {list(vdf.columns)}")

    vdf = vdf.sort_values("frame_idx").reset_index(drop=True)

    frame_idx = vdf["frame_idx"].to_numpy(dtype=np.int64)
    t_abs = vdf["t_sec"].to_numpy(dtype=np.float64)

    if len(t_abs) < 2 or (not np.all(np.isfinite(t_abs))):
        raise RuntimeError(f"Invalid t_sec in {ts_csv_path} (len={len(t_abs)})")

    # sanity: frame_idx should start at 0 and be monotonic increasing
    if frame_idx[0] != 0:
        print(f"[WARN] frame_idx does not start at 0 (starts at {frame_idx[0]}). "
              f"Will still index by decode order i -> t_abs[i].")

    # sanity: check continuity (optional)
    d = np.diff(frame_idx)
    if not np.all(d == 1):
        bad = np.where(d != 1)[0][:10]
        print(f"[WARN] frame_idx not contiguous in {ts_csv_path}. Example break positions: {bad.tolist()} "
              f"(this may indicate dropped rows). Proceeding anyway.")

    return t_abs


@click.command()
@click.option('-i', '--input', required=True, help='Input raw_video.mp4 path')
@click.option('-o', '--output', required=True, help='Output tag_detection.pkl path')
@click.option('-ij', '--intrinsics_json', required=True, help='Camera intrinsics json file')
@click.option('-ay', '--aruco_yaml', required=True, help='Aruco config yaml file')
@click.option('-n', '--num_workers', type=int, default=4, help='OpenCV threads / decoder threads')
@click.option('--timestamps_csv', default=None,
              help='Optional path to raw_video_timestamps.csv. If not set, use input dir/raw_video_timestamps.csv')
@click.option('--save-pts-time/--no-save-pts-time', default=True,
              help='Save pts-based time as pts_time for debugging (default: True)')
def main(input, output, intrinsics_json, aruco_yaml, num_workers, timestamps_csv, save_pts_time):
    """
    Detect ArUco tags for each frame in raw_video.mp4 and save results to tag_detection.pkl.

    FINAL BEHAVIOR:
      - result["time"] is ABSOLUTE epoch seconds from raw_video_timestamps.csv (NOT pts*time_base).
      - Optional: result["pts_time"] is pts-based time for debugging.
    """
    cv2.setNumThreads(num_workers)

    input_path = os.path.expanduser(input)
    output_path = os.path.expanduser(output)

    # --- load aruco config
    aruco_config = parse_aruco_config(yaml.safe_load(open(aruco_yaml, 'r')))
    aruco_dict = aruco_config['aruco_dict']
    marker_size_map = aruco_config['marker_size_map']

    # --- load intrinsics
    raw_fisheye_intr = parse_fisheye_intrinsics(json.load(open(intrinsics_json, 'r')))

    # --- load absolute per-frame timestamps (epoch seconds)
    input_dir = os.path.dirname(input_path)
    ts_csv_path = os.path.expanduser(timestamps_csv) if timestamps_csv else os.path.join(input_dir, "raw_video_timestamps.csv")
    if not os.path.isfile(ts_csv_path):
        raise RuntimeError(f"raw_video_timestamps.csv not found next to raw_video.mp4: {ts_csv_path}")

    t_abs = _load_video_timestamps_csv(ts_csv_path)

    results = []

    with av.open(input_path) as in_container:
        in_stream = in_container.streams.video[0]
        in_stream.thread_type = "AUTO"
        in_stream.thread_count = num_workers

        in_res = np.array([in_stream.height, in_stream.width])[::-1]
        fisheye_intr = convert_fisheye_intrinsics_resolution(
            opencv_intr_dict=raw_fisheye_intr, target_resolution=in_res
        )

        total_frames_hint = int(in_stream.frames) if (in_stream.frames is not None and int(in_stream.frames) > 0) else len(t_abs)

        for i, frame in tqdm(enumerate(in_container.decode(in_stream)),
                             total=total_frames_hint,
                             desc=os.path.basename(input_path)):
            if i >= len(t_abs):
                raise RuntimeError(
                    f"Decoded more frames than timestamps: decoded i={i} but len(t_abs)={len(t_abs)}. "
                    f"Check raw_video_timestamps.csv generation vs mp4."
                )

            img = frame.to_ndarray(format='rgb24')
            img = draw_predefined_mask(img, color=(0, 0, 0), mirror=True, gripper=False, finger=False)

            tag_dict = detect_localize_aruco_tags(
                img=img,
                aruco_dict=aruco_dict,
                marker_size_map=marker_size_map,
                fisheye_intr_dict=fisheye_intr,
                refine_subpix=True
            )

            result = {
                'frame_idx': i,
                'time': float(t_abs[i]),          # ABS epoch seconds
                'tag_dict': tag_dict,
                'time_source': 'raw_video_timestamps.csv'
            }

            if save_pts_time:
                pts_time = None
                if frame.pts is not None and in_stream.time_base is not None:
                    pts_time = float(frame.pts) * float(in_stream.time_base)
                result['pts_time'] = pts_time

            results.append(result)

    if len(results) != len(t_abs):
        print(f"[WARN] decoded frames ({len(results)}) != timestamps rows ({len(t_abs)}). "
              f"Downstream should truncate to min length.")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved tag detections to: {output_path}")


if __name__ == "__main__":
    main()