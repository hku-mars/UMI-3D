#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run UMI-3D dataset pipeline for an aligned_bags/session directory.

Stages:
  00_detect_aruco
  01_run_calibrations
  02_generate_dataset_plan
  03_generate_replay_buffer

Default calibration files:
  example/calibration/fisheye.json
  example/calibration/aruco_config.yaml

Example:
python run_dataset_pipeline.py \
  --session_dir /path/to/aligned_bags \
  --output /path/to/aligned_bags/dataset_door_01.zarr.zip

Optional override:
python run_dataset_pipeline.py \
  --session_dir /path/to/aligned_bags \
  --output /path/to/aligned_bags/dataset_door_01.zarr.zip \
  --camera_intrinsics /path/to/custom_fisheye.json \
  --aruco_config /path/to/custom_aruco_config.yaml
"""

import os
import sys
import pathlib
import subprocess
import click


ROOT_DIR = pathlib.Path(__file__).resolve().parent
SCRIPT_DIR = ROOT_DIR / "scripts_slam_pipeline"
DEFAULT_CALIB_DIR = ROOT_DIR / "example" / "calibration"


def run_cmd(cmd):
    print("\n[RUN]", " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"Command failed with return code {result.returncode}")


@click.command()
@click.option(
    "--session_dir",
    required=True,
    type=str,
    help="Path to aligned_bags root (must contain demos/)."
)
@click.option(
    "--output",
    required=True,
    type=str,
    help="Output zarr zip path, e.g. /path/to/aligned_bags/dataset_door_01.zarr.zip"
)
@click.option(
    "--camera_intrinsics",
    default=None,
    type=str,
    help="Optional path to fisheye.json. Default: example/calibration/fisheye.json"
)
@click.option(
    "--aruco_config",
    default=None,
    type=str,
    help="Optional path to aruco_config.yaml. Default: example/calibration/aruco_config.yaml"
)
@click.option(
    "--num_workers",
    default=None,
    type=int,
    help="Optional worker count for 00_detect_aruco.py"
)
@click.option(
    "--force_aruco",
    is_flag=True,
    default=False,
    help="Force regenerate tag_detection.pkl even if it already exists"
)
@click.option(
    "--out_res",
    default="224,224",
    type=str,
    help="Output image resolution for replay buffer, e.g. 224,224"
)
@click.option(
    "--compression_level",
    default=99,
    type=int,
    help="Compression level for replay buffer"
)
@click.option(
    "--no_mirror",
    is_flag=True,
    default=False,
    help="Pass --no_mirror to 03_generate_replay_buffer.py"
)
@click.option(
    "--mirror_swap",
    is_flag=True,
    default=False,
    help="Pass --mirror_swap to 03_generate_replay_buffer.py"
)
def main(
    session_dir,
    output,
    camera_intrinsics,
    aruco_config,
    num_workers,
    force_aruco,
    out_res,
    compression_level,
    no_mirror,
    mirror_swap,
):
    session_dir = pathlib.Path(os.path.expanduser(session_dir)).resolve()
    demos_dir = session_dir / "demos"
    output = pathlib.Path(os.path.expanduser(output)).resolve()

    if camera_intrinsics is None:
        camera_intrinsics = DEFAULT_CALIB_DIR / "fisheye.json"
    else:
        camera_intrinsics = pathlib.Path(os.path.expanduser(camera_intrinsics)).resolve()

    if aruco_config is None:
        aruco_config = DEFAULT_CALIB_DIR / "aruco_config.yaml"
    else:
        aruco_config = pathlib.Path(os.path.expanduser(aruco_config)).resolve()

    # ---- basic checks
    if not SCRIPT_DIR.is_dir():
        raise SystemExit(f"scripts_slam_pipeline not found: {SCRIPT_DIR}")

    required_scripts = [
        SCRIPT_DIR / "00_detect_aruco.py",
        SCRIPT_DIR / "01_run_calibrations.py",
        SCRIPT_DIR / "02_generate_dataset_plan.py",
        SCRIPT_DIR / "03_generate_replay_buffer.py",
    ]
    for sp in required_scripts:
        if not sp.is_file():
            raise SystemExit(f"Required script not found: {sp}")

    if not session_dir.is_dir():
        raise SystemExit(f"session_dir not found: {session_dir}")

    if not demos_dir.is_dir():
        raise SystemExit(f"demos dir not found under session_dir: {demos_dir}")

    if not camera_intrinsics.is_file():
        raise SystemExit(f"camera_intrinsics not found: {camera_intrinsics}")

    if not aruco_config.is_file():
        raise SystemExit(f"aruco_config not found: {aruco_config}")

    output.parent.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    print("====================================================")
    print("UMI-3D Dataset Pipeline")
    print("ROOT_DIR          =", ROOT_DIR)
    print("SESSION_DIR       =", session_dir)
    print("DEMOS_DIR         =", demos_dir)
    print("CAMERA_INTRINSICS =", camera_intrinsics)
    print("ARUCO_CONFIG      =", aruco_config)
    print("OUTPUT            =", output)
    print("====================================================")

    # 00_detect_aruco.py
    print("\n############# 00_detect_aruco ###########")
    cmd = [
        py, str(SCRIPT_DIR / "00_detect_aruco.py"),
        "-i", str(demos_dir),
        "-ci", str(camera_intrinsics),
        "-ac", str(aruco_config),
    ]
    if num_workers is not None:
        cmd += ["-n", str(num_workers)]
    if force_aruco:
        cmd += ["--force"]
    run_cmd(cmd)

    # 01_run_calibrations.py
    print("\n############# 01_run_calibrations ###########")
    cmd = [
        py, str(SCRIPT_DIR / "01_run_calibrations.py"),
        str(session_dir)
    ]
    run_cmd(cmd)

    # 02_generate_dataset_plan.py
    print("\n############# 02_generate_dataset_plan ###########")
    cmd = [
        py, str(SCRIPT_DIR / "02_generate_dataset_plan.py"),
        "-i", str(session_dir)
    ]
    run_cmd(cmd)

    dataset_plan_path = session_dir / "dataset_plan.pkl"
    if not dataset_plan_path.is_file():
        raise SystemExit(f"dataset_plan.pkl not found after stage 02: {dataset_plan_path}")

    # 03_generate_replay_buffer.py
    print("\n############# 03_generate_replay_buffer ###########")
    cmd = [
        py, str(SCRIPT_DIR / "03_generate_replay_buffer.py"),
        str(session_dir),
        "-o", str(output),
        "-or", str(out_res),
        "-cl", str(compression_level),
    ]
    if no_mirror:
        cmd += ["-nm"]
    if mirror_swap:
        cmd += ["-ms"]
    run_cmd(cmd)

    if not output.is_file():
        raise SystemExit(f"Replay buffer output not found after stage 03: {output}")

    print("\n====================================================")
    print("Done.")
    print("dataset plan  :", dataset_plan_path)
    print("replay buffer :", output)
    print("====================================================")


if __name__ == "__main__":
    main()
