#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
auto_bag_to_mp4_aligned_clean.py

Goal:
  - Do NOT rename original *.bag
  - (Optional) Strictly align cam/lidar/imu into aligned_bags/ with fixed-width ids
  - Export MP4 videos from aligned bag (or original bag if --align not set)
  - (Optional) organize each video into demos/.../raw_video*.mp4
  - NEW: save per-frame timestamps for exported video into CSV sidecar file

STRICT ALIGNMENT (image:lidar = 2:1):
  For each LiDAR frame k with time tl:
    anchor0 = tl
    anchor1 = tl + 0.5 * T_lidar
  Find TWO nearest images (monotonic consume) within --gate seconds.
  If either anchor fails, drop this LiDAR frame.

Time source:
  - default: rosbag record time 't'
  - recommended for your Livox + camera data: --use_header_stamp

Recommended command (your verified good setup):
  python3 auto_bag_to_mp4_aligned.py \
    --dir . --align --organize_each \
    --start_idx 0 --id_width 6 \
    --use_header_stamp --gate 0.02 \
    --no_symlink

Notes:
  - On /media/... external disk (exFAT/NTFS), symlink often fails. Use --no_symlink (default).
  - If your output is on ext4 and you want symlink, use --try_symlink

CHANGE (2026-02-21):
  - gripper_calibration* demos now live together with other demos under ONE demos/ folder.
    i.e. in --align mode:
      aligned_bags/demos/demo_000001_000001/...
      aligned_bags/demos/gripper_calibration_xxx/...
    instead of:
      aligned_bags/gripper_calibration_xxx/demos/gripper_calibration_xxx/...
"""

import os
import glob
import argparse
import shutil
import subprocess
import re
import hashlib
import csv
from datetime import datetime
from typing import Optional, List, Set, Tuple, Dict

import rosbag
import cv2
from cv_bridge import CvBridge
from tqdm import tqdm


# ------------------------- config -------------------------

PREFERRED_TOPICS = [
    "/left_camera/image",
    "/camera/image_raw",
    "/rgb/image",
    "/image",
]

DEFAULT_CAM_TOPIC = "/left_camera/image"
DEFAULT_LIDAR_TOPIC = "/livox/lidar"
DEFAULT_IMU_TOPIC = "/livox/imu"

TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})-(\d{2}-\d{2}-\d{2})")


# ------------------------- small utils -------------------------

def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def format_id(num: int, width: int) -> str:
    return f"{int(num):0{int(width)}d}"


def bag_sort_key(path: str):
    """
    Sort rule:
      1) if filename has YYYY-MM-DD-HH-MM-SS timestamp -> sort by that
      2) else sort by mtime
    """
    base = os.path.basename(path)
    m = TS_RE.search(base)
    if m:
        ts_str = f"{m.group(1)}-{m.group(2)}"
        try:
            dt = datetime.strptime(ts_str, "%Y-%m-%d-%H-%M-%S")
            return (0, dt.timestamp(), base)
        except Exception:
            pass
    return (1, os.path.getmtime(path), base)


def is_gripper_calib_bag(bag_path: str) -> bool:
    base = os.path.splitext(os.path.basename(bag_path))[0]
    return base.startswith("gripper_calibration")


def gripper_calib_folder_name(bag_path: str) -> str:
    return os.path.splitext(os.path.basename(bag_path))[0]


def _short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def _rel_symlink_target(src_path: str, dst_path: str) -> str:
    try:
        return os.path.relpath(dst_path, start=os.path.dirname(src_path))
    except Exception:
        return dst_path


# ------------------------- alignment core (time helper) -------------------------

def _get_msg_time(msg, t, use_header_stamp: bool):
    if use_header_stamp and hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        return msg.header.stamp
    return t


# ------------------------- video export helpers -------------------------

def list_image_topics(bag_path: str) -> List[str]:
    with rosbag.Bag(bag_path, "r") as bag:
        info = bag.get_type_and_topic_info()
        topics = []
        for topic_name, topic_info in info.topics.items():
            if topic_info.msg_type == "sensor_msgs/Image":
                topics.append(topic_name)
        return sorted(topics)


def choose_topic(topics: List[str]) -> Optional[str]:
    if not topics:
        return None
    if len(topics) == 1:
        return topics[0]
    for pref in PREFERRED_TOPICS:
        if pref in topics:
            return pref
    return topics[0]


def estimate_fps(ts_list) -> float:
    """Estimate fps from timestamps (median dt)."""
    if len(ts_list) < 2:
        return 30.0
    diffs = []
    for i in range(1, len(ts_list)):
        dt = (ts_list[i] - ts_list[i - 1]).to_sec()
        if dt > 1e-6:
            diffs.append(dt)
    if not diffs:
        return 30.0
    diffs.sort()
    median_dt = diffs[len(diffs) // 2]
    fps = 1.0 / median_dt if median_dt > 0 else 30.0
    return max(1.0, min(120.0, fps))


def msg_to_bgr(bridge: CvBridge, msg):
    try:
        return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except Exception:
        img = bridge.imgmsg_to_cv2(msg)
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def count_topic_msgs(bag_path: str, topic: str) -> int:
    cnt = 0
    with rosbag.Bag(bag_path, "r") as bag:
        for _ in bag.read_messages(topics=[topic]):
            cnt += 1
    return cnt


def ffmpeg_avi_to_mp4(avi_path: str, mp4_path: str):
    cmd = [
        "ffmpeg", "-y", "-i", avi_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        mp4_path
    ]
    subprocess.check_call(cmd)


def write_frame_timestamps_csv(csv_path: str, rows: List[Tuple[int, float, int, int]]):
    """
    rows: (frame_idx, t_sec, t_nsec, use_header_stamp_int)
    """
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "t_sec", "t_nsec", "use_header_stamp"])
        for r in rows:
            w.writerow(list(r))


def write_avi_from_bag(
    bag_path: str,
    topic: str,
    avi_path: str,
    fps_force: float = 0.0,
    use_header_stamp: bool = False,
):
    """
    Returns:
      n_frames, fps, (w,h), ts_rows
    where ts_rows is list of (frame_idx, t_sec, t_nsec, use_header_stamp_int)
    """
    bridge = CvBridge()

    ts = []
    first_msg = None
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            ts.append(_get_msg_time(msg, t, use_header_stamp))
            if first_msg is None:
                first_msg = msg
            if len(ts) >= 200:
                break

    if first_msg is None:
        raise RuntimeError(f"No frames found on topic {topic}")

    fps = fps_force if (fps_force and fps_force > 0) else estimate_fps(ts)

    frame0 = msg_to_bgr(bridge, first_msg)
    h, w = frame0.shape[:2]
    size = (w, h)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(avi_path, fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {avi_path}")

    total_frames = count_topic_msgs(bag_path, topic)

    ts_rows: List[Tuple[int, float, int, int]] = []
    n = 0
    with rosbag.Bag(bag_path, "r") as bag:
        it = bag.read_messages(topics=[topic])
        for _, msg, t in tqdm(it, total=total_frames, desc=os.path.basename(bag_path), unit="frame"):
            frame = msg_to_bgr(bridge, msg)
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            writer.write(frame)

            ts_msg = _get_msg_time(msg, t, use_header_stamp)
            sec = float(ts_msg.to_sec())
            # ros::Time has nsecs in ROS1; fallback if absent
            if hasattr(ts_msg, "nsecs"):
                nsec = int(ts_msg.nsecs)
            else:
                nsec = int(round((sec - float(int(sec))) * 1e9))
            ts_rows.append((n, sec, nsec, 1 if use_header_stamp else 0))
            n += 1

    writer.release()
    return n, fps, size, ts_rows


def export_mp4_from_bag(
    bag_path: str,
    fps_force: float = 0.0,
    keep_avi: bool = False,
    use_header_stamp: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Export MP4 from the chosen image topic.
    Output under <bag_dir>/raw_videos/<bag_base>.mp4
    NEW: also output <bag_dir>/raw_videos/<bag_base>_frame_timestamps.csv
    Returns: (mp4_path, ts_csv_path)
    """
    bag_path = os.path.abspath(bag_path)
    bag_dir = os.path.dirname(bag_path)
    bag_base = os.path.splitext(os.path.basename(bag_path))[0]

    out_dir = os.path.join(bag_dir, "raw_videos")
    ensure_dir(out_dir)

    topics = list_image_topics(bag_path)
    if not topics:
        print(f"[SKIP] {os.path.basename(bag_path)}: no sensor_msgs/Image topics")
        return None, None

    topic = choose_topic(topics)
    if len(topics) > 1:
        print(f"[INFO] {os.path.basename(bag_path)}: image topics = {topics}")
        print(f"[INFO] {os.path.basename(bag_path)}: chosen topic = {topic}")

    avi_path = os.path.join(out_dir, f"{bag_base}.avi")
    mp4_path = os.path.join(out_dir, f"{bag_base}.mp4")
    ts_csv_path = os.path.join(out_dir, f"{bag_base}_frame_timestamps.csv")

    n, fps, size, ts_rows = write_avi_from_bag(
        bag_path, topic, avi_path,
        fps_force=fps_force,
        use_header_stamp=use_header_stamp,
    )
    print(f"[OK] {bag_base}: wrote {n} frames -> {avi_path} (fps={fps:.2f}, size={size[0]}x{size[1]})")

    write_frame_timestamps_csv(ts_csv_path, ts_rows)
    print(f"[OK] {bag_base}: timestamps -> {ts_csv_path} (use_header_stamp={use_header_stamp})")

    if shutil.which("ffmpeg") is None:
        print("[WARN] ffmpeg not found; keep AVI only.")
        return None, ts_csv_path

    ffmpeg_avi_to_mp4(avi_path, mp4_path)
    print(f"[OK] {bag_base}: converted -> {mp4_path}")

    if not keep_avi:
        try:
            os.remove(avi_path)
        except OSError:
            pass

    return mp4_path, ts_csv_path


# ------------------------- alignment core (STRICT start+mid) -------------------------

def collect_timestamps(bag_path: str, cam_topic: str, lidar_topic: str, use_header_stamp: bool) -> Tuple[list, list]:
    cam_ts = []
    lidar_ts = []
    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[cam_topic, lidar_topic]):
            ts = _get_msg_time(msg, t, use_header_stamp)
            if topic == cam_topic:
                cam_ts.append(ts)
            elif topic == lidar_topic:
                lidar_ts.append(ts)
    return cam_ts, lidar_ts


def estimate_period(ts_list) -> Optional[float]:
    if len(ts_list) < 2:
        return None
    diffs = []
    for i in range(1, len(ts_list)):
        dt = (ts_list[i] - ts_list[i - 1]).to_sec()
        if dt > 1e-6:
            diffs.append(dt)
    if not diffs:
        return None
    diffs.sort()
    return diffs[len(diffs) // 2]


def median_float(sorted_vals: List[float]) -> float:
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])


def percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    q = max(0.0, min(1.0, q))
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def nearest_index(cam_ts, target_sec: float, j0: int) -> int:
    """
    cam_ts: increasing
    find nearest index >= j0 around insertion point
    """
    n = len(cam_ts)
    if j0 >= n:
        return -1

    j = j0
    while j < n and cam_ts[j].to_sec() < target_sec:
        j += 1

    candidates = []
    if j < n:
        candidates.append(j)
    if j - 1 >= j0:
        candidates.append(j - 1)

    if not candidates:
        return -1

    best = candidates[0]
    best_dt = abs(cam_ts[best].to_sec() - target_sec)
    for idx in candidates[1:]:
        dt = abs(cam_ts[idx].to_sec() - target_sec)
        if dt < best_dt:
            best_dt = dt
            best = idx
    return best


def match_strict_2x_start_mid(
    cam_ts,
    lidar_ts,
    gate_sec: float,
    T_lidar: float,
) -> Tuple[Set[int], Set[int], List[float], List[Tuple[int,int,float,float]]]:
    """
    Returns:
      lidar_keep (indices)
      cam_keep (indices)
      dt_list (sec, for both anchors)
      pair_debug: list of (lidar_i, idx0, idx1, dt0, dt1) for accepted frames
    """
    nC = len(cam_ts)
    nL = len(lidar_ts)
    cam_keep: Set[int] = set()
    lidar_keep: Set[int] = set()
    dt_list: List[float] = []
    pair_dbg: List[Tuple[int,int,int,float,float]] = []

    if nC < 2 or nL < 1:
        return set(), set(), [], []

    j = 0
    for li in range(nL):
        tl = lidar_ts[li].to_sec()
        t0 = tl
        t1 = tl + 0.5 * T_lidar

        idx0 = nearest_index(cam_ts, t0, j0=j)
        if idx0 < 0:
            break

        idx1 = nearest_index(cam_ts, t1, j0=idx0 + 1)
        if idx1 < 0:
            break

        dt0 = abs(cam_ts[idx0].to_sec() - t0)
        dt1 = abs(cam_ts[idx1].to_sec() - t1)

        if dt0 > gate_sec or dt1 > gate_sec:
            continue

        lidar_keep.add(li)
        cam_keep.add(idx0)
        cam_keep.add(idx1)
        dt_list.append(dt0)
        dt_list.append(dt1)
        pair_dbg.append((li, idx0, idx1, dt0, dt1))

        j = idx1 + 1
        if j >= nC:
            break

    return lidar_keep, cam_keep, dt_list, pair_dbg


def write_aligned_bag_by_index_set(
    in_bag_path: str,
    out_bag_path: str,
    cam_topic: str,
    lidar_topic: str,
    imu_topic: str,
    cam_keep: Set[int],
    lidar_keep: Set[int],
    imu_margin_sec: float,
) -> Tuple[Dict[str,int], Tuple[float,float]]:
    # IMU window computed from kept LiDAR using bag time 't'
    lidar_ts_all = []
    with rosbag.Bag(in_bag_path, "r") as bag:
        for _, _, t in bag.read_messages(topics=[lidar_topic]):
            lidar_ts_all.append(t)

    if not lidar_keep:
        raise RuntimeError("No kept lidar frames after strict matching")

    kept = sorted(lidar_keep)
    t0 = lidar_ts_all[kept[0]]
    t1 = lidar_ts_all[kept[-1]]
    t0_sec = t0.to_sec() - float(imu_margin_sec)
    t1_sec = t1.to_sec() + float(imu_margin_sec)

    cam_idx = 0
    lidar_idx = 0
    wrote = {cam_topic: 0, lidar_topic: 0, imu_topic: 0}

    with rosbag.Bag(in_bag_path, "r") as inbag, rosbag.Bag(out_bag_path, "w") as outbag:
        for topic, msg, t in inbag.read_messages(topics=[cam_topic, lidar_topic, imu_topic]):
            if topic == cam_topic:
                if cam_idx in cam_keep:
                    outbag.write(topic, msg, t)
                    wrote[cam_topic] += 1
                cam_idx += 1
            elif topic == lidar_topic:
                if lidar_idx in lidar_keep:
                    outbag.write(topic, msg, t)
                    wrote[lidar_topic] += 1
                lidar_idx += 1
            elif topic == imu_topic:
                ts = t.to_sec()
                if t0_sec <= ts <= t1_sec:
                    outbag.write(topic, msg, t)
                    wrote[imu_topic] += 1

    return wrote, (t0_sec, t1_sec)


def align_one_bag(
    bag_path: str,
    cam_topic: str,
    lidar_topic: str,
    imu_topic: str,
    out_dir: str,
    out_name: str,
    gate_sec: float,
    max_median_dt_sec: float,
    imu_margin_sec: float,
    use_header_stamp: bool,
    lidar_period_override: float,
    debug_keep: bool,
    dry_run: bool,
):
    ensure_dir(out_dir)
    out_bag_path = os.path.join(out_dir, out_name)
    out_id = os.path.splitext(out_name)[0]
    orig_name = os.path.basename(bag_path)

    cam_ts, lidar_ts = collect_timestamps(bag_path, cam_topic, lidar_topic, use_header_stamp=use_header_stamp)
    cam_n = len(cam_ts)
    lidar_n = len(lidar_ts)
    if cam_n < 2 or lidar_n < 1:
        print(f"[ALIGN][SKIP] {orig_name}: insufficient frames cam={cam_n} lidar={lidar_n}")
        return None, None

    T = float(lidar_period_override) if (lidar_period_override and lidar_period_override > 0) else estimate_period(lidar_ts)
    if T is None or T <= 0:
        T = 0.1  # fallback 10Hz

    lidar_keep, cam_keep, dt_list, pair_dbg = match_strict_2x_start_mid(
        cam_ts=cam_ts, lidar_ts=lidar_ts, gate_sec=gate_sec, T_lidar=T
    )

    keep_lidar = len(lidar_keep)
    keep_cam = len(cam_keep)
    if keep_lidar == 0 or keep_cam != 2 * keep_lidar:
        print(f"[ALIGN][SKIP] {orig_name}: strict matching failed. keep_lidar={keep_lidar}, keep_cam={keep_cam}")
        return None, None

    dt_sorted = sorted(dt_list)
    med_dt = median_float(dt_sorted)
    p95_dt = percentile(dt_sorted, 0.95)
    max_dt = dt_sorted[-1] if dt_sorted else float("nan")

    print(f"[ALIGN] {orig_name} -> {os.path.basename(out_bag_path)}")
    print(f"        cam_frames={cam_n} lidar_frames={lidar_n}")
    print(f"        STRICT start+mid, image:lidar=2:1, gate={gate_sec*1000:.1f} ms, use_header_stamp={use_header_stamp}")
    print(f"        lidar_period≈{T*1000:.2f} ms (mid=+{0.5*T*1000:.2f} ms)")
    print(f"        dt_stats(ms): median={med_dt*1000:.2f}, p95={p95_dt*1000:.2f}, max={max_dt*1000:.2f}")
    print(f"        keep_lidar={keep_lidar}, keep_cam={keep_cam} (ratio OK)")

    if debug_keep and pair_dbg:
        first = pair_dbg[0]
        last = pair_dbg[-1]
        li0, c00, c01, d00, d01 = first
        li1, c10, c11, d10, d11 = last
        print(f"[DEBUG] first_keep: lidar_idx={li0}, cam_idx=({c00},{c01}), dt(ms)=({d00*1000:.3f},{d01*1000:.3f})")
        print(f"[DEBUG] last_keep : lidar_idx={li1}, cam_idx=({c10},{c11}), dt(ms)=({d10*1000:.3f},{d11*1000:.3f})")

    if med_dt > max_median_dt_sec:
        print(f"[ALIGN][WARN] median_dt {med_dt:.6f}s > {max_median_dt_sec:.6f}s (still proceeding)")

    if dry_run:
        print(f"[DRY_RUN] would write aligned bag -> {out_bag_path}")
        wrote = {cam_topic: keep_cam, lidar_topic: keep_lidar, imu_topic: -1}
        imu_win = (float("nan"), float("nan"))
    else:
        wrote, imu_win = write_aligned_bag_by_index_set(
            in_bag_path=bag_path,
            out_bag_path=out_bag_path,
            cam_topic=cam_topic,
            lidar_topic=lidar_topic,
            imu_topic=imu_topic,
            cam_keep=cam_keep,
            lidar_keep=lidar_keep,
            imu_margin_sec=imu_margin_sec,
        )
        print(f"[ALIGN][OK] wrote aligned bag -> {out_bag_path}")
        print(f"           wrote: cam={wrote[cam_topic]}, lidar={wrote[lidar_topic]}, imu={wrote[imu_topic]}")

    stats = {
        "id": out_id,
        "aligned_bag": os.path.basename(out_bag_path),
        "orig_bag": orig_name,
        "out_dir": os.path.abspath(out_dir),
        "gate_ms": float(gate_sec * 1000.0),
        "median_abs_dt_ms": float(med_dt * 1000.0),
        "p95_dt_ms": float(p95_dt * 1000.0),
        "max_dt_ms": float(max_dt * 1000.0),
        "lidar_period_ms": float(T * 1000.0),
        "keep_lidar": int(keep_lidar),
        "keep_cam": int(keep_cam),
        "cam_n": int(cam_n),
        "lidar_n": int(lidar_n),
        "cam_written": int(wrote.get(cam_topic, 0)),
        "lidar_written": int(wrote.get(lidar_topic, 0)),
        "imu_written": int(wrote.get(imu_topic, 0)),
        "imu_win": (float(imu_win[0]), float(imu_win[1])),
        "use_header_stamp": bool(use_header_stamp),
        "mode": "strict_start_mid",
    }
    return out_bag_path, stats


def write_aligned_index_txt(index_path: str, rows: List[dict]):
    header = [
        "id", "aligned_bag", "orig_bag", "out_dir",
        "mode", "gate_ms", "median_abs_dt_ms", "p95_dt_ms", "max_dt_ms",
        "lidar_period_ms", "keep_lidar", "keep_cam", "cam_n", "lidar_n",
        "cam_written", "lidar_written", "imu_written",
        "imu_t0", "imu_t1", "use_header_stamp",
    ]
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for s in rows:
            imu_t0, imu_t1 = s["imu_win"]
            line = [
                str(s["id"]),
                str(s["aligned_bag"]),
                str(s["orig_bag"]),
                str(s.get("out_dir", "")),
                str(s.get("mode", "")),
                f"{float(s.get('gate_ms', 0.0)):.3f}",
                f"{float(s.get('median_abs_dt_ms', 0.0)):.3f}",
                f"{float(s.get('p95_dt_ms', 0.0)):.3f}",
                f"{float(s.get('max_dt_ms', 0.0)):.3f}",
                f"{float(s.get('lidar_period_ms', 0.0)):.3f}",
                str(s["keep_lidar"]),
                str(s["keep_cam"]),
                str(s["cam_n"]),
                str(s["lidar_n"]),
                str(s["cam_written"]),
                str(s["lidar_written"]),
                str(s["imu_written"]),
                f"{imu_t0:.6f}",
                f"{imu_t1:.6f}",
                str(int(bool(s.get("use_header_stamp", False)))),
            ]
            f.write("\t".join(line) + "\n")


# ------------------------- organizer helpers -------------------------

def ensure_demo_dir(session_dir: str, demo_id_str: str, prefix: str) -> str:
    session_dir = os.path.abspath(session_dir)
    demos_dir = os.path.join(session_dir, "demos")
    ensure_dir(demos_dir)
    demo_name = f"{prefix}_{demo_id_str}_{demo_id_str}"
    out_dir = os.path.join(demos_dir, demo_name)
    ensure_dir(out_dir)
    return out_dir


def ensure_gripper_demo_dir(session_dir: str, bag_base: str) -> str:
    session_dir = os.path.abspath(session_dir)
    root = os.path.join(session_dir, "demos", bag_base)
    ensure_dir(root)
    return root


def move_or_copy(src: str, dst: str, copy: bool):
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def organize_video_with_timestamps(
    mp4_path: str,
    ts_csv_path: Optional[str],
    demo_dir: str,
    copy: bool,
    no_symlink: bool,
    try_symlink: bool,
) -> Tuple[str, Optional[str]]:
    """
    Move/copy mp4 into demo_dir as raw_video*.mp4
    Also move/copy timestamps CSV into demo_dir as raw_video_timestamps*.csv
    """
    ensure_dir(demo_dir)

    out_video = os.path.join(demo_dir, "raw_video.mp4")
    suffix = ""
    if os.path.exists(out_video):
        suffix = _short_hash(mp4_path)
        out_video = os.path.join(demo_dir, f"raw_video_{suffix}.mp4")

    move_or_copy(mp4_path, out_video, copy=copy)

    # default: no symlink (safe for exFAT/NTFS)
    if (not no_symlink) and (not copy) and try_symlink:
        target = _rel_symlink_target(mp4_path, out_video)
        try:
            os.symlink(target, mp4_path)
        except Exception as e:
            print(f"[ORG][WARN] symlink failed: {e}")

    out_ts = None
    if ts_csv_path is not None and os.path.exists(ts_csv_path):
        out_ts = os.path.join(demo_dir, "raw_video_timestamps.csv")
        if suffix:
            out_ts = os.path.join(demo_dir, f"raw_video_timestamps_{suffix}.csv")
        move_or_copy(ts_csv_path, out_ts, copy=copy)

    return out_video, out_ts


def write_source_txt(demo_dir: str, demo_id: str, orig_bag: str, is_gripper: bool, stats: Optional[dict]):
    meta_path = os.path.join(demo_dir, "source.txt")
    if os.path.exists(meta_path):
        return
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"demo_id:    {demo_id}\n")
            f.write(f"orig_bag:   {orig_bag}\n")
            if is_gripper:
                f.write("type:       gripper_calibration\n")
            if stats is not None:
                f.write(f"aligned_bag:{stats.get('aligned_bag','')}\n")
                f.write(f"mode:      {stats.get('mode','')}\n")
                f.write(f"gate_ms:   {stats.get('gate_ms','')}\n")
                f.write(f"median_dt_ms:{stats.get('median_abs_dt_ms','')}\n")
                f.write(f"p95_dt_ms: {stats.get('p95_dt_ms','')}\n")
                f.write(f"max_dt_ms: {stats.get('max_dt_ms','')}\n")
                f.write(f"lidar_period_ms:{stats.get('lidar_period_ms','')}\n")
                f.write(f"keep_lidar:{stats.get('keep_lidar','')}\n")
                f.write(f"keep_cam:  {stats.get('keep_cam','')}\n")
                f.write(f"use_header_stamp:{int(bool(stats.get('use_header_stamp', False)))}\n")
    except Exception:
        pass


# ------------------------- main -------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Clean strict start+mid bag aligner + mp4 exporter + demo organizer (+timestamps CSV)"
    )

    # basic
    p.add_argument("--dir", default=".", help="Directory to scan for *.bag (default: current)")
    p.add_argument("--dry_run", action="store_true", help="Print what would happen, without writing bags/videos")
    p.add_argument("--debug_keep", action="store_true", help="Print first/last kept lidar->cam match stats")

    # video export
    p.add_argument("--fps", type=float, default=0.0, help="Force fps for output video (0=auto from timestamps)")
    p.add_argument("--keep_avi", action="store_true", help="Keep intermediate AVI files")

    # naming
    p.add_argument("--start_idx", type=int, default=0, help="Start index for normal bags in aligned_bags")
    p.add_argument("--id_width", type=int, default=6, help="Zero-pad width for aligned ids (default 6)")

    # align
    p.add_argument("--align", action="store_true", help="Enable strict alignment and write aligned bags")
    p.add_argument("--cam_topic", default=DEFAULT_CAM_TOPIC)
    p.add_argument("--lidar_topic", default=DEFAULT_LIDAR_TOPIC)
    p.add_argument("--imu_topic", default=DEFAULT_IMU_TOPIC)
    p.add_argument("--aligned_dir", default="aligned_bags", help="Output subdir for aligned outputs")

    # strict knobs (recommend your tested)
    p.add_argument("--use_header_stamp", action="store_true", help="Use msg.header.stamp for matching (recommended)")
    p.add_argument("--gate", type=float, default=0.02, help="Strict gate seconds (recommended 0.02 for your data)")
    p.add_argument("--lidar_period", type=float, default=0.0, help="Override LiDAR period seconds (0=auto median)")
    p.add_argument("--max_median_dt", type=float, default=0.05, help="Warn if median abs dt > this seconds")
    p.add_argument("--imu_margin", type=float, default=0.0, help="IMU time window extra margin seconds")

    # organize
    p.add_argument("--organize_each", action="store_true", help="Move/copy mp4 into demos/... immediately")
    p.add_argument("--prefix", default="demo", help="Demo folder prefix (default demo)")
    p.add_argument("--copy", action="store_true", help="Copy videos into demos/ (non-destructive)")

    # symlink behavior:
    p.add_argument("--no_symlink", action="store_true", help="Do not create symlink (recommended for /media external disks)")
    p.add_argument("--try_symlink", action="store_true", help="Try to create symlink at original mp4 location (only if filesystem supports)")

    # gripper
    p.add_argument("--gripper_force_numeric", action="store_true",
                   help="If set, gripper_calibration*.bag will also be named as 000xxx in its folder")

    return p


def main():
    args = build_parser().parse_args()
    target_dir = os.path.abspath(args.dir)
    id_width = int(args.id_width)

    bag_list = sorted(glob.glob(os.path.join(target_dir, "*.bag")), key=bag_sort_key)
    if not bag_list:
        print(f"[DONE] No .bag files found in {target_dir}")
        return

    print(f"[INFO] Found {len(bag_list)} bag files in {target_dir}")
    print(f"[INFO] Original bags will NOT be renamed.")

    aligned_root = os.path.join(target_dir, args.aligned_dir)
    ensure_dir(aligned_root)

    # demos 的统一根目录：
    # - --align: demos 放到 aligned_root/demos 下（gripper 与 demo_* 同级）
    # - no --align: demos 放到 target_dir/demos 下
    demo_root_for_session = aligned_root if args.align else target_dir

    # if user didn't specify, default is safe no symlink
    no_symlink = args.no_symlink or (not args.try_symlink)

    if args.align:
        print(f"[INFO] STRICT start+mid enabled: image:lidar=2:1")
        print(f"       use_header_stamp={args.use_header_stamp}, gate={args.gate*1000:.1f} ms, lidar_period_override={args.lidar_period}")
        print(f"       output aligned_root={aligned_root}")
        print(f"       demos_root={demo_root_for_session}  (ALL demos under one folder)")
    if args.organize_each:
        print(f"[INFO] Organize videos into demos/: copy={args.copy}, no_symlink={no_symlink}, try_symlink={args.try_symlink}")

    index_rows = []
    normal_counter = 0

    for bag_path in bag_list:
        base = os.path.splitext(os.path.basename(bag_path))[0]
        is_gripper = is_gripper_calib_bag(bag_path)

        # Decide out_dir/out_name/demo_id
        if is_gripper:
            sub = gripper_calib_folder_name(bag_path)
            out_dir = os.path.join(aligned_root, sub)
            ensure_dir(out_dir)
            if args.gripper_force_numeric:
                demo_id = format_id(args.start_idx + normal_counter, id_width)
                out_name = f"{demo_id}.bag"
            else:
                demo_id = base
                out_name = f"{base}.bag"
        else:
            out_dir = aligned_root
            demo_id = format_id(args.start_idx + normal_counter, id_width)
            out_name = f"{demo_id}.bag"
            normal_counter += 1

        bag_for_video = bag_path
        stats = None

        try:
            # Align if needed
            if args.align:
                aligned, stats = align_one_bag(
                    bag_path=bag_path,
                    cam_topic=args.cam_topic,
                    lidar_topic=args.lidar_topic,
                    imu_topic=args.imu_topic,
                    out_dir=out_dir,
                    out_name=out_name,
                    gate_sec=float(args.gate),
                    max_median_dt_sec=float(args.max_median_dt),
                    imu_margin_sec=float(args.imu_margin),
                    use_header_stamp=bool(args.use_header_stamp),
                    lidar_period_override=float(args.lidar_period),
                    debug_keep=bool(args.debug_keep),
                    dry_run=bool(args.dry_run),
                )
                if aligned is None:
                    print(f"[SKIP] {os.path.basename(bag_path)}: alignment failed")
                    continue
                bag_for_video = aligned
                index_rows.append(stats)

            # Create demo dir and source.txt  (IMPORTANT CHANGE: use demo_root_for_session)
            if is_gripper:
                demo_dir = ensure_gripper_demo_dir(session_dir=demo_root_for_session, bag_base=base)
            else:
                demo_dir = ensure_demo_dir(session_dir=demo_root_for_session, demo_id_str=demo_id, prefix=args.prefix)

            write_source_txt(demo_dir, demo_id, os.path.basename(bag_path), is_gripper, stats)

            # Export video
            if args.dry_run:
                print(f"[DRY_RUN] would export mp4 from {os.path.basename(bag_for_video)}")
                continue

            mp4_path, ts_csv_path = export_mp4_from_bag(
                bag_for_video,
                fps_force=args.fps,
                keep_avi=args.keep_avi,
                use_header_stamp=bool(args.use_header_stamp),
            )

            # Organize
            if args.organize_each and mp4_path is not None:
                out_video, out_ts = organize_video_with_timestamps(
                    mp4_path=mp4_path,
                    ts_csv_path=ts_csv_path,
                    demo_dir=demo_dir,
                    copy=args.copy,
                    no_symlink=no_symlink,
                    try_symlink=args.try_symlink,
                )
                if out_ts is not None:
                    print(f"[ORG][OK] {os.path.basename(out_video)} + {os.path.basename(out_ts)} saved in {demo_dir}")
                else:
                    print(f"[ORG][OK] {os.path.basename(out_video)} saved in {demo_dir}")

        except Exception as e:
            print(f"[ERROR] {os.path.basename(bag_path)}: {e}")

    # Write index.txt
    if args.align and (not args.dry_run):
        index_path = os.path.join(aligned_root, "index.txt")
        write_aligned_index_txt(index_path, index_rows)
        print(f"[INDEX][OK] wrote -> {index_path} ({len(index_rows)} rows)")


if __name__ == "__main__":
    main()
