#!/usr/bin/env bash
set -euo pipefail

log()  { echo -e "\033[1;32m[INFO]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m $*" >&2; }

# -------------------------
# auto-detect repo/workspace
# -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# UMI-3D-SLAM naming
UMI_3D_SLAM_ROOT="${UMI_3D_SLAM_ROOT:-$REPO_ROOT/umi_3d_slam_ws}"
SLAM_PKG_NAME="${SLAM_PKG_NAME:-umi_3d_slam}"
UMI_3D_SLAM_CSV="${UMI_3D_SLAM_CSV:-$UMI_3D_SLAM_ROOT/src/$SLAM_PKG_NAME/output/camera_trajectory.csv}"

UMI_3D_SLAM_LAUNCH_FILE="${UMI_3D_SLAM_LAUNCH_FILE:-mapping_mid360_180_min.launch}"
UMI_3D_SLAM_LAUNCH_ARGS="${UMI_3D_SLAM_LAUNCH_ARGS:-rviz:=false}"

# defaults
BAG_DIR=""
DEMOS_DIR=""
MODE=""
START_ID=""
END_ID=""
IDS=()

ROSBAG_ARGS="${ROSBAG_ARGS:---wait-for-subscribers --queue=5000}"
WAIT_CSV_TIMEOUT_SEC="${WAIT_CSV_TIMEOUT_SEC:-30}"
WAIT_STABLE_SEC="${WAIT_STABLE_SEC:-2.0}"
CHECK_PERIOD_SEC="${CHECK_PERIOD_SEC:-0.2}"
MAX_STABLE_WAIT_SEC="${MAX_STABLE_WAIT_SEC:-60}"
MIN_START_LINES="${MIN_START_LINES:-6}"
WAIT_NODE_TIMEOUT_SEC="${WAIT_NODE_TIMEOUT_SEC:-15}"

# ⚠️ 保持默认节点名，避免破坏运行
UMI_3D_SLAM_NODE_NAME="${UMI_3D_SLAM_NODE_NAME:-/laserMapping}"

ID_WIDTH="${ID_WIDTH:-6}"
SKIP_PREFIX="${SKIP_PREFIX:-gripper_calibration}"
DELETE_BAG_AFTER_SUCCESS="${DELETE_BAG_AFTER_SUCCESS:-1}"

usage() {
  cat <<EOF
Usage (UMI-3D-SLAM batch pipeline):

  Range mode:
    $0 --bag_dir /path/to/aligned_bags --start 0 --end 100

  List mode:
    $0 --bag_dir /path/to/aligned_bags --list 0 5 10

Optional:
    --demos_dir /path/to/demos

Defaults:
  demos_dir = <bag_dir>/demos

Environment overrides:
  UMI_3D_SLAM_ROOT, SLAM_PKG_NAME, UMI_3D_SLAM_CSV
  UMI_3D_SLAM_LAUNCH_FILE, UMI_3D_SLAM_LAUNCH_ARGS, UMI_3D_SLAM_NODE_NAME
EOF
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { err "Missing command: $1"; exit 1; }
}

normalize_id() {
  local raw="$1"
  [[ "$raw" =~ ^[0-9]+$ ]] || { err "Invalid id: $raw"; return 1; }
  python3 - <<PY
w=int("${ID_WIDTH}")
n=int("${raw}")
print(f"{n:0{w}d}")
PY
}

file_mtime() {
  [[ -f "$1" ]] && stat -c %Y "$1" 2>/dev/null || echo 0
}

file_lines() {
  [[ -f "$1" ]] && wc -l < "$1" | tr -d ' ' || echo 0
}

cleanup_ros() {
  warn "Cleaning ROS processes"
  rosnode kill -a >/dev/null 2>&1 || true

  # ⚠️ 保持原进程名
  pkill -f fastlivo_mapping >/dev/null 2>&1 || true
  pkill -f save_camera_trajectory >/dev/null 2>&1 || true
  pkill -f "rosbag play" >/dev/null 2>&1 || true

  sleep 1.0
}

wait_rosnode() {
  local name="$1"
  local timeout="$2"
  local t0=$(date +%s)

  while true; do
    rosnode list 2>/dev/null | grep -qx "$name" && return 0
    (( $(date +%s) - t0 >= timeout )) && return 1
    sleep 0.2
  done
}

wait_file_fresh() {
  local file="$1"
  local start_ts="$2"
  local timeout="$3"
  local t0=$(date +%s)

  while true; do
    [[ -f "$file" ]] && [[ $(file_mtime "$file") -ge $start_ts ]] && return 0
    (( $(date +%s) - t0 >= timeout )) && return 1
    sleep 0.2
  done
}

run_one_bag() {
  local id=$(normalize_id "$1")

  local bag="$BAG_DIR/${id}.bag"
  local demo_dir="$DEMOS_DIR/demo_${id}_${id}"
  local dst_csv="$demo_dir/camera_trajectory.csv"
  local log_dir="$demo_dir/umi_3d_slam_logs"

  [[ -f "$bag" ]] || { err "Bag not found: $bag"; return 2; }
  mkdir -p "$log_dir"

  log "Processing bag $id with UMI-3D-SLAM"
  cleanup_ros

  local start_ts=$(date +%s)
  [[ -f "$UMI_3D_SLAM_CSV" ]] && rm -f "$UMI_3D_SLAM_CSV"

  log "Launching UMI-3D-SLAM..."
  (
    cd "$UMI_3D_SLAM_ROOT"
    ./umi_3d_slam.sh
  ) >"$log_dir/roslaunch.log" 2>&1 &

  local slam_pid=$!

  if ! wait_rosnode "$UMI_3D_SLAM_NODE_NAME" "$WAIT_NODE_TIMEOUT_SEC"; then
    err "SLAM node not started"
    kill "$slam_pid" >/dev/null 2>&1 || true
    return 3
  fi

  rosbag play $ROSBAG_ARGS "$bag" >"$log_dir/rosbag.log" 2>&1 || true

  rosnode kill -a >/dev/null 2>&1 || true
  kill "$slam_pid" >/dev/null 2>&1 || true

  if ! wait_file_fresh "$UMI_3D_SLAM_CSV" "$start_ts" "$WAIT_CSV_TIMEOUT_SEC"; then
    err "Trajectory not generated"
    return 4
  fi

  mv -f "$UMI_3D_SLAM_CSV" "$dst_csv"

  [[ -s "$dst_csv" ]] && rm -f "$bag"

  log "Done: $id"
}

# -------------------------
# parse args
# -------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bag_dir) BAG_DIR="$2"; shift 2;;
    --demos_dir) DEMOS_DIR="$2"; shift 2;;
    --start) START_ID="$2"; shift 2;;
    --end) END_ID="$2"; shift 2;;
    --list)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        IDS+=("$1"); shift
      done
      MODE="list"
      ;;
    *) err "Unknown argument: $1"; usage; exit 1;;
  esac
done

[[ -n "$BAG_DIR" ]] || { err "--bag_dir required"; exit 1; }
BAG_DIR="$(cd "$BAG_DIR" && pwd)"
DEMOS_DIR="${DEMOS_DIR:-$BAG_DIR/demos}"

if [[ "$MODE" != "list" ]]; then
  for ((i=START_ID;i<=END_ID;i++)); do IDS+=("$i"); done
fi

log "UMI-3D-SLAM ROOT: $UMI_3D_SLAM_ROOT"
log "Bag dir: $BAG_DIR"

for id in "${IDS[@]}"; do
  run_one_bag "$id"
done

log "All done 🚀"
