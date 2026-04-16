#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UMI-3D single-gripper dataset plan generator (FINAL VERSION)
- Main timeline uses raw_video_timestamps.csv (absolute epoch seconds per frame)
- tag_detection.pkl is aligned by frame index (sanity check: tag.time ~= video time)
- camera_trajectory.csv is aligned to video frames by timestamp via merge_asof (nearest within tolerance)
- Robust to inconsistent fps metadata, missing pose frames, CSV NaNs, and partial tracking loss

Run:
python scripts_slam_pipeline/06_generate_dataset_plan.py \
  -i /path/to/session_or_aligned_bags
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import pickle
import numpy as np
import json
import collections
import pandas as pd
import av
from exiftool import ExifToolHelper
from scipy.spatial.transform import Rotation

from umi.common.timecode_util import mp4_get_start_datetime
from umi.common.pose_util import pose_to_mat, mat_to_pose
from umi.common.cv_util import get_gripper_width
from umi.common.interpolation_util import (
    get_gripper_calibration_interpolator,
    get_interp1d,
)

# ---------------------------------------------------------------------
# Alignment knobs
# ---------------------------------------------------------------------

# Pose timestamp alignment tolerance (seconds)
POSE_ALIGN_TOL_SEC = 0.01  # 10ms (your logs show max_abs ~ 3.5ms, so 10ms is safe & stricter)

# If tag_time and video_time differ more than this, warn (seconds)
TAG_VIDEO_WARN_SEC = 0.005  # 5ms sanity-check threshold

# If pose alignment quality is below this, print more debug
POSE_MATCH_WARN_RATIO = 0.99  # 99%

# Print top-K worst dt samples when debugging
DEBUG_TOPK = 10


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_bool_segments(bool_seq):
    bool_seq = np.array(bool_seq, dtype=bool)
    segment_ends = (np.nonzero(np.diff(bool_seq))[0] + 1).tolist()
    segment_bounds = [0] + segment_ends + [len(bool_seq)]
    segments = []
    segment_type = []
    for i in range(len(segment_bounds) - 1):
        start = segment_bounds[i]
        end = segment_bounds[i + 1]
        this_type = bool_seq[start]
        segments.append(slice(start, end))
        segment_type.append(this_type)
    return segments, np.array(segment_type, dtype=bool)


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _print_dt_stats(name, ts: np.ndarray):
    if ts is None or len(ts) < 2:
        print(f"[WARN] {name}: timestamps length < 2")
        return
    d = np.diff(ts.astype(np.float64))
    good = d[(d > 1e-6) & np.isfinite(d)]
    if len(good) == 0:
        print(f"[WARN] {name}: no valid dt in timestamps (diff<=0 or nan)")
        return
    print(f"[DT] {name}: median={np.median(good):.6f}s  min={np.min(good):.6f}s  max={np.max(good):.6f}s  n={len(ts)}")


def _to_bool_series(x: pd.Series) -> pd.Series:
    """Robust bool conversion for is_lost style columns."""
    if x.dtype == bool:
        return x
    return x.astype(str).str.lower().isin(['true', '1', 't', 'yes'])


def _require_cols(df: pd.DataFrame, cols, context: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[WARN] {context}: missing columns {missing}")
        return False
    return True


def _debug_print_worst_alignment(df_aligned: pd.DataFrame, tol_sec: float, name: str, topk: int = 10):
    """
    df_aligned must have columns: video_frame_idx, t_video, t_pose
    Print worst |dt| among matched rows, and sample unmatched rows.
    """
    # matched mask
    m = np.isfinite(df_aligned['t_pose'].to_numpy(dtype=np.float64))
    if np.any(m):
        dt = (df_aligned.loc[m, 't_video'].to_numpy(dtype=np.float64) -
              df_aligned.loc[m, 't_pose'].to_numpy(dtype=np.float64))
        absdt = np.abs(dt)
        idx = np.argsort(-absdt)[:topk]
        sub = df_aligned.loc[df_aligned.loc[m].index[idx], ['video_frame_idx', 't_video', 't_pose']].copy()
        sub['dt'] = (sub['t_video'] - sub['t_pose']).astype(np.float64)
        print(f"[DEBUG] {name}: worst matched dt samples (top {min(topk, len(sub))}), tol={tol_sec:.3f}s")
        for r in sub.itertuples(index=False):
            print(f"        frame={int(r.video_frame_idx):6d}  dt={float(r.dt): .6f}s  "
                  f"t_video={float(r.t_video):.6f}  t_pose={float(r.t_pose):.6f}")

    # unmatched
    um = ~m
    if np.any(um):
        um_idx = np.where(um)[0][:topk].tolist()
        print(f"[DEBUG] {name}: unmatched pose frames (show first {len(um_idx)}):")
        for i in um_idx:
            r = df_aligned.iloc[i]
            print(f"        frame={int(r['video_frame_idx']):6d}  t_video={float(r['t_video']):.6f}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

@click.command()
@click.option('-i', '--input', required=True, help='Session directory (contains demos/) or aligned_bags dir')
@click.option('-o', '--output', default=None, help='Output dataset_plan.pkl path')
@click.option('-to', '--tcp_offset', type=float, default=0.205,
              help="Distance from gripper tip to mounting screw (meters)")
@click.option('-nz', '--nominal_z', type=float, default=0.072,
              help="Nominal Z value for gripper finger tag (meters)")
@click.option('-ml', '--min_episode_length', type=int, default=24,
              help="Minimum episode length in frames")
def main(input, output, tcp_offset, nominal_z, min_episode_length):
    # ---- stage 0: paths
    input_path = pathlib.Path(os.path.expanduser(input)).absolute()

    demos_dir = input_path.joinpath('demos')
    if not demos_dir.is_dir():
        if input_path.name == 'demos' and input_path.is_dir():
            demos_dir = input_path
            input_path = demos_dir.parent
        else:
            raise RuntimeError(f"demos dir not found: {demos_dir} (input={input_path})")

    if output is None:
        output_path = input_path.joinpath('dataset_plan.pkl')
    else:
        output_path = pathlib.Path(os.path.expanduser(output)).absolute()

    # ---- tcp to camera transform (UMI-3D params)
    cam_to_center_height = 0.086   # meters
    cam_to_mount_offset = 0.01465  # meters
    cam_to_tip_offset = cam_to_mount_offset + tcp_offset
    pose_cam_tcp = np.array([0, cam_to_center_height, cam_to_tip_offset, 0, 0, 0], dtype=float)
    tx_cam_tcp = pose_to_mat(pose_cam_tcp)

    # ---- base frame is SLAM frame (identity)
    tx_base_slam = np.eye(4, dtype=float)

    # ---- load gripper calibration(s)
    gripper_id_gripper_cal_map = {}
    cam_serial_gripper_cal_map = {}

    gripper_range_paths = list(demos_dir.glob("gripper*/gripper_range.json"))
    if len(gripper_range_paths) == 0:
        raise RuntimeError(f"No gripper_range.json found under {demos_dir}/gripper*")

    with ExifToolHelper() as et:
        for gripper_range_path in gripper_range_paths:
            mp4_path = gripper_range_path.parent.joinpath('raw_video.mp4')
            if not mp4_path.is_file():
                print(f"[WARN] {mp4_path} not found, skip camera-serial map")
                cam_serial = "unknown"
            else:
                meta = list(et.get_metadata(str(mp4_path)))[0]
                cam_serial = meta.get('QuickTime:CameraSerialNumber', "unknown")

            gripper_range_data = json.load(gripper_range_path.open('r'))
            gripper_id = int(gripper_range_data['gripper_id'])
            max_width = float(gripper_range_data['max_width'])
            min_width = float(gripper_range_data['min_width'])

            gripper_cal_data = {
                'aruco_measured_width': [min_width, max_width],
                'aruco_actual_width': [min_width, max_width]
            }
            gripper_cal_interp = get_gripper_calibration_interpolator(**gripper_cal_data)
            gripper_id_gripper_cal_map[gripper_id] = gripper_cal_interp
            cam_serial_gripper_cal_map[cam_serial] = gripper_cal_interp

    # ---- stage 1: scan demo videos and metadata
    video_dirs = sorted([x.parent for x in demos_dir.glob('demo_*/raw_video.mp4')])
    if len(video_dirs) == 0:
        raise RuntimeError(f"No demo_*/raw_video.mp4 found under {demos_dir}")

    rows = []
    fps0 = None
    with ExifToolHelper() as et:
        for video_dir in video_dirs:
            mp4_path = video_dir.joinpath('raw_video.mp4')
            meta = list(et.get_metadata(str(mp4_path)))[0]
            cam_serial = meta.get('QuickTime:CameraSerialNumber', "unknown")

            # mp4 start datetime (coarse; only for bookkeeping)
            start_date = mp4_get_start_datetime(str(mp4_path))
            start_timestamp = start_date.timestamp()

            csv_path = video_dir.joinpath('camera_trajectory.csv')
            pkl_path = video_dir.joinpath('tag_detection.pkl')
            vts_path = video_dir.joinpath('raw_video_timestamps.csv')

            if not csv_path.is_file():
                print(f"[WARN] Ignored {video_dir.name}: no camera_trajectory.csv")
                continue
            if not pkl_path.is_file():
                print(f"[WARN] Ignored {video_dir.name}: no tag_detection.pkl")
                continue
            if not vts_path.is_file():
                print(f"[WARN] Ignored {video_dir.name}: no raw_video_timestamps.csv")
                continue

            with av.open(str(mp4_path), 'r') as container:
                stream = container.streams.video[0]
                n_frames = stream.frames
                this_fps = _safe_float(stream.average_rate, default=None)
                if fps0 is None:
                    fps0 = this_fps
                else:
                    if (this_fps is not None) and (fps0 is not None) and abs(this_fps - float(fps0)) > 0.1:
                        print(f"[WARN] Inconsistent fps(meta): {float(fps0):.6f} vs {this_fps:.6f} in {video_dir.name} "
                              f"(OK, we use raw_video_timestamps.csv)")

            # coarse end_timestamp (bookkeeping)
            if (this_fps is None) or (this_fps <= 1e-6) or (n_frames is None) or (n_frames <= 0):
                duration_sec = 0.0
            else:
                duration_sec = float(n_frames / float(this_fps))
            end_timestamp = start_timestamp + duration_sec

            rows.append({
                'video_dir': video_dir,
                'camera_serial': cam_serial,
                'start_date': start_date,
                'n_frames': int(n_frames) if n_frames is not None else -1,
                'fps': this_fps if this_fps is not None else fps0,
                'start_timestamp': float(start_timestamp),
                'end_timestamp': float(end_timestamp)
            })

    if len(rows) == 0:
        raise RuntimeError("No valid demo videos (missing csv/pkl/raw_video_timestamps?)")

    video_meta_df = pd.DataFrame(rows)

    # ---- stage 2: each video is one demo window
    demo_data_list = []
    for vid_idx, row in video_meta_df.iterrows():
        demo_data_list.append({
            "video_idxs": [vid_idx],
            "start_timestamp": float(row['start_timestamp']),
            "end_timestamp": float(row['end_timestamp'])
        })

    # ---- stage 3: identify gripper hardware id from tags
    finger_tag_det_th = 0.8
    vid_idx_gripper_hardware_id_map = {}
    cam_serial_gripper_ids_map = collections.defaultdict(list)

    for vid_idx, row in video_meta_df.iterrows():
        video_dir = row['video_dir']
        pkl_path = video_dir.joinpath('tag_detection.pkl')
        tag_data = pickle.load(pkl_path.open('rb'))

        n_frames = len(tag_data)
        tag_counts = collections.defaultdict(lambda: 0)
        for frame in tag_data:
            for key in frame.get('tag_dict', {}).keys():
                tag_counts[key] += 1

        tag_stats = collections.defaultdict(lambda: 0.0)
        for k, v in tag_counts.items():
            tag_stats[int(k)] = v / max(1, n_frames)

        max_tag_id = int(np.max(list(tag_stats.keys()))) if len(tag_stats) > 0 else -1
        tag_per_gripper = 6
        max_gripper_id = max_tag_id // tag_per_gripper if max_tag_id >= 0 else -1

        gripper_prob_map = {}
        for gripper_id in range(max_gripper_id + 1):
            left_id = gripper_id * tag_per_gripper
            right_id = left_id + 1
            left_prob = tag_stats[left_id]
            right_prob = tag_stats[right_id]
            gripper_prob = min(left_prob, right_prob)
            if gripper_prob > 0:
                gripper_prob_map[gripper_id] = gripper_prob

        gripper_id_by_tag = -1
        if len(gripper_prob_map) > 0:
            gripper_id, gripper_prob = sorted(gripper_prob_map.items(), key=lambda x: x[-1])[-1]
            if gripper_prob >= finger_tag_det_th:
                gripper_id_by_tag = int(gripper_id)

        cam_serial_gripper_ids_map[row['camera_serial']].append(gripper_id_by_tag)
        vid_idx_gripper_hardware_id_map[vid_idx] = gripper_id_by_tag

    video_meta_df['gripper_hardware_id'] = pd.Series(vid_idx_gripper_hardware_id_map)

    # ---- stage 4: force single gripper camera_idx = 0
    video_meta_df['camera_idx'] = 0
    print("Assigned camera_idx: single gripper = 0")
    print(video_meta_df[['video_dir', 'camera_serial', 'camera_idx', 'gripper_hardware_id']])

    # ---- stage 6: generate dataset plan
    total_available_time = 0.0
    total_used_time = 0.0
    dropped_camera_count = collections.defaultdict(lambda: 0)
    n_dropped_demos = 0
    all_plans = []

    for demo_idx, demo_data in enumerate(demo_data_list):
        video_idxs = demo_data['video_idxs']
        demo_video_meta_df = video_meta_df.loc[video_idxs].copy()
        demo_video_meta_df.set_index('camera_idx', inplace=True)
        demo_video_meta_df.sort_index(inplace=True)

        row = demo_video_meta_df.iloc[0]
        video_dir = row['video_dir']
        csv_path = video_dir.joinpath('camera_trajectory.csv')
        pkl_path = video_dir.joinpath('tag_detection.pkl')
        video_ts_path = video_dir.joinpath('raw_video_timestamps.csv')

        # ---- load video absolute timestamps (main timeline)
        vts_df = pd.read_csv(video_ts_path)
        if not _require_cols(vts_df, ['frame_idx', 't_sec'], f"{video_dir.name}: raw_video_timestamps.csv"):
            print(f"[WARN] Skipped {video_dir.name}: raw_video_timestamps.csv missing required cols")
            n_dropped_demos += 1
            continue

        vts_df = vts_df.sort_values('frame_idx').reset_index(drop=True)
        t_video_full = vts_df['t_sec'].to_numpy(dtype=np.float64)

        if (len(t_video_full) < 2) or (not np.all(np.isfinite(t_video_full))):
            print(f"[WARN] Skipped {video_dir.name}: invalid t_sec in raw_video_timestamps.csv")
            n_dropped_demos += 1
            continue

        _print_dt_stats(video_dir.name + " [video_ts]", t_video_full)

        # ---- load tag detections (frame-aligned with video by index)
        tag_detection_results_full = pickle.load(open(pkl_path, 'rb'))
        if len(tag_detection_results_full) < 2:
            print(f"[WARN] Skipped {video_dir.name}: tag_detection too short")
            n_dropped_demos += 1
            continue

        # ---- common length by (video_ts, tag_detection)
        N = min(len(t_video_full), len(tag_detection_results_full))
        if N < 2:
            print(f"[WARN] Skipped {video_dir.name}: too short after align "
                  f"(video_ts={len(t_video_full)}, tag={len(tag_detection_results_full)})")
            n_dropped_demos += 1
            continue

        if len(t_video_full) != len(tag_detection_results_full):
            print(f"[WARN] {video_dir.name}: video_ts/tag length mismatch "
                  f"(video_ts={len(t_video_full)}, tag={len(tag_detection_results_full)}), trunc to {N}")

        t_video = t_video_full[:N]
        tag_detection_results = tag_detection_results_full[:N]

        # ---- sanity check: tag time should match video time
        tag_t = np.array([x.get('time', np.nan) for x in tag_detection_results], dtype=np.float64)
        if np.all(np.isfinite(tag_t)):
            diff = tag_t - t_video
            max_abs = float(np.max(np.abs(diff)))
            med = float(np.median(diff))
            print(f"[CHECK] {video_dir.name}: tag_time - video_time median={med:.9f}s  max_abs={max_abs:.9f}s")
            if max_abs > TAG_VIDEO_WARN_SEC:
                print(f"[WARN] {video_dir.name}: tag_time not aligned to raw_video_timestamps "
                      f"(>{TAG_VIDEO_WARN_SEC*1000:.1f}ms). Using t_video as ground truth timeline.")
        else:
            print(f"[WARN] {video_dir.name}: invalid tag time; ignoring tag time and using t_video only.")

        # ---- available window by video timestamps (ground truth)
        start_timestamp = float(t_video[0])
        end_timestamp = float(t_video[-1])
        total_available_time += max(0.0, (end_timestamp - start_timestamp))

        # ---- load pose CSV
        csv_df_full = pd.read_csv(csv_path)
        if 'timestamp' not in csv_df_full.columns:
            print(f"[WARN] Skipped {video_dir.name}: camera_trajectory.csv missing 'timestamp' col")
            n_dropped_demos += 1
            continue

        if 'is_lost' in csv_df_full.columns:
            csv_df_full['is_lost'] = _to_bool_series(csv_df_full['is_lost'])
        else:
            csv_df_full['is_lost'] = False

        # ---- align pose rows to video frames by absolute timestamp (merge_asof)
        cdf = csv_df_full.copy()
        cdf['t_pose'] = cdf['timestamp'].astype(np.float64)
        cdf = cdf[np.isfinite(cdf['t_pose'])].sort_values('t_pose').reset_index(drop=True)

        vdf = pd.DataFrame({
            'video_frame_idx': np.arange(N, dtype=int),
            't_video': t_video.astype(np.float64)
        }).sort_values('t_video').reset_index(drop=True)

        aligned = pd.merge_asof(
            vdf,
            cdf,
            left_on='t_video',
            right_on='t_pose',
            direction='nearest',
            tolerance=POSE_ALIGN_TOL_SEC
        )

        # restore original order by frame index
        df = aligned.sort_values('video_frame_idx').reset_index(drop=True)

        # report alignment quality
        matched_ratio = float(np.isfinite(df['t_pose']).mean())
        dt_pose = (df['t_video'] - df['t_pose']).to_numpy(dtype=np.float64)
        dt_pose = dt_pose[np.isfinite(dt_pose)]

        if len(dt_pose) > 0:
            print(f"[ALIGN] {video_dir.name}: pose matched={matched_ratio*100:.1f}%  "
                  f"dt median={float(np.median(dt_pose)):.6f}s  max_abs={float(np.max(np.abs(dt_pose))):.6f}s  "
                  f"tol={POSE_ALIGN_TOL_SEC:.3f}s")
        else:
            print(f"[ALIGN] {video_dir.name}: pose matched={matched_ratio*100:.1f}%  (no dt stats)  "
                  f"tol={POSE_ALIGN_TOL_SEC:.3f}s")

        if (matched_ratio < POSE_MATCH_WARN_RATIO) or (len(dt_pose) > 0 and float(np.max(np.abs(dt_pose))) > 0.8 * POSE_ALIGN_TOL_SEC):
            _debug_print_worst_alignment(df, POSE_ALIGN_TOL_SEC, video_dir.name, topk=DEBUG_TOPK)

        # ---- FIX: robust tracking mask & NaN pose handling
        pose_cols = ['x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w']
        if not _require_cols(df, pose_cols, f"{video_dir.name}: aligned pose df"):
            print(f"[WARN] Skipped {video_dir.name}: missing pose columns after alignment")
            n_dropped_demos += 1
            continue

        pose_nan_mask = df[pose_cols].isna().any(axis=1).to_numpy(dtype=bool)

        # no match (t_pose NaN) -> lost
        df['is_lost'] = df['is_lost'].fillna(True)

        # any NaN pose -> lost
        df.loc[pose_nan_mask, 'is_lost'] = True

        is_tracked = (~df['is_lost'].to_numpy(dtype=bool))

        # fill identity for lost+NaN pose to avoid NaNs in Rotation
        lost_mask = df['is_lost'].to_numpy(dtype=bool)
        fix_mask = lost_mask & pose_nan_mask
        if np.any(fix_mask):
            df.loc[fix_mask, ['x', 'y', 'z']] = 0.0
            df.loc[fix_mask, ['q_x', 'q_y', 'q_z']] = 0.0
            df.loc[fix_mask, ['q_w']] = 1.0

        # ---- drop if too many lost frames (same policy as before)
        n_frames_lost = int((~is_tracked).sum())
        if n_frames_lost > 10:
            print(f"Skipping {video_dir.name}, {n_frames_lost} frames are lost.")
            dropped_camera_count[row['camera_serial']] += 1
            n_dropped_demos += 1
            continue

        n_frames_valid = int(is_tracked.sum())
        if n_frames_valid < 1:
            print(f"Skipping {video_dir.name}, only {n_frames_valid} frames are valid.")
            dropped_camera_count[row['camera_serial']] += 1
            n_dropped_demos += 1
            continue

        # ---- build camera pose
        cam_pos = df[['x', 'y', 'z']].to_numpy(dtype=np.float64)
        cam_rot_quat_xyzw = df[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy(dtype=np.float64)

        q_norm = np.linalg.norm(cam_rot_quat_xyzw, axis=1)
        bad = np.where(~np.isfinite(q_norm) | (q_norm < 1e-8))[0]
        if len(bad) > 0:
            print("\n================ BAD QUAT DETECTED ================")
            print("video_dir:", video_dir)
            print("csv_path :", csv_path)
            print("bad indices:", bad.tolist())
            cols = [c for c in ['frame_idx', 'timestamp', 'is_lost', 'have_odom',
                                'q_x', 'q_y', 'q_z', 'q_w', 'x', 'y', 'z'] if c in df.columns]
            print(df.iloc[bad[:10]][cols])
            print("===================================================\n")
            print(f"[WARN] Dropping {video_dir.name} due to invalid quaternion (not recoverable safely).")
            n_dropped_demos += 1
            continue

        cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)

        cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
        cam_pose[:, 3, 3] = 1
        cam_pose[:, :3, 3] = cam_pos
        cam_pose[:, :3, :3] = cam_rot.as_matrix()

        tx_slam_cam = cam_pose
        tx_base_cam = tx_base_slam @ tx_slam_cam

        # ---- demo timeline = video absolute timestamps (ground truth)
        demo_timestamps = t_video.astype(np.float64).copy()
        diffs = np.diff(demo_timestamps)
        diffs = diffs[(diffs > 1e-6) & np.isfinite(diffs)]
        dt_med = float(np.median(diffs)) if len(diffs) > 0 else 0.0

        # ---- gripper width
        ghi = int(row['gripper_hardware_id'])
        if ghi < 0:
            print(f"Skipping {video_dir.name}, invalid gripper hardware id {ghi}")
            dropped_camera_count[row['camera_serial']] += 1
            n_dropped_demos += 1
            continue

        left_id = 6 * ghi
        right_id = left_id + 1

        if ghi in gripper_id_gripper_cal_map:
            gripper_cal_interp = gripper_id_gripper_cal_map[ghi]
        elif row['camera_serial'] in cam_serial_gripper_cal_map:
            gripper_cal_interp = cam_serial_gripper_cal_map[row['camera_serial']]
            print(f"[WARN] Gripper id {ghi} not found in gripper calibrations. Falling back to camera serial map.")
        else:
            raise RuntimeError("Gripper calibration not found.")

        gripper_timestamps = []
        gripper_widths = []
        for td in tag_detection_results:
            width = get_gripper_width(
                td.get('tag_dict', {}),
                left_id=left_id,
                right_id=right_id,
                nominal_z=nominal_z
            )
            if width is not None:
                # td['time'] is absolute epoch seconds (after detect_aruco.py update)
                gripper_timestamps.append(float(td['time']))
                gripper_widths.append(float(gripper_cal_interp(width)))

        if len(gripper_timestamps) < 2:
            print(f"Skipping {video_dir.name}, not enough gripper detections.")
            n_dropped_demos += 1
            continue

        gripper_interp = get_interp1d(gripper_timestamps, gripper_widths)
        det_ratio = len(gripper_widths) / max(1, len(tag_detection_results))
        if det_ratio < 0.9:
            print(f"[WARN] {video_dir.name}: only {det_ratio:.3f} of frames have gripper tags detected.")

        this_gripper_widths = gripper_interp(demo_timestamps)

        # ---- tcp pose = cam pose * tx_cam_tcp
        tx_base_tcp = tx_base_cam @ tx_cam_tcp
        pose_base_tcp = mat_to_pose(tx_base_tcp)

        # ---- valid mask from tracking
        is_step_valid = is_tracked.copy()

        # remove segments that are too short
        segment_slices, segment_type = get_bool_segments(is_step_valid)
        for s, is_valid_segment in zip(segment_slices, segment_type):
            if not is_valid_segment:
                continue
            if (s.stop - s.start) < min_episode_length:
                is_step_valid[s.start:s.stop] = False

        # generate episodes per valid segment
        segment_slices, segment_type = get_bool_segments(is_step_valid)
        for s, is_valid in zip(segment_slices, segment_type):
            if not is_valid:
                continue
            start = s.start
            end = s.stop

            # time accounting: use real timestamps (video timeline)
            if end - start >= 2:
                total_used_time += float(demo_timestamps[end - 1] - demo_timestamps[start])
            elif dt_med > 0:
                total_used_time += float((end - start) * dt_med)

            grippers = [{
                "tcp_pose": pose_base_tcp[start:end],
                "gripper_width": this_gripper_widths[start:end],
                "demo_start_pose": pose_base_tcp[start],
                "demo_end_pose": pose_base_tcp[end - 1],
            }]

            cameras = [{
                "video_path": str(video_dir.joinpath('raw_video.mp4').relative_to(video_dir.parent)),
                "video_start_end": (start, end)
            }]

            all_plans.append({
                "episode_timestamps": demo_timestamps[start:end],
                "grippers": grippers,
                "cameras": cameras
            })

    used_ratio = (total_used_time / total_available_time) if total_available_time > 0 else 0.0
    print(f"{int(used_ratio * 100)}% of raw data are used.")
    print("dropped_camera_count:", dict(dropped_camera_count))
    print("n_dropped_demos:", n_dropped_demos)
    print("n_episodes:", len(all_plans))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('wb') as f:
        pickle.dump(all_plans, f)

    print(f"Saved dataset plan to: {output_path}")


if __name__ == "__main__":
    main()