#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch run detect_aruco.py for all demo_*/raw_video.mp4 under input_dir.

Example:
python scripts_slam_pipeline/04_detect_aruco.py \
  -i data_workspace/cup_in_the_wild/xxx/demos \
  -ci data_workspace/toss_objects/20231113/calibration/gopro_intrinsics_2_7k.json \
  -ac data_workspace/toss_objects/20231113/calibration/aruco_config.yaml
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import multiprocessing
import subprocess
import concurrent.futures
from tqdm import tqdm


@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for demos folder')
@click.option('-ci', '--camera_intrinsics', required=True, help='Camera intrinsics json file (2.7k)')
@click.option('-ac', '--aruco_yaml', required=True, help='Aruco config yaml file')
@click.option('-n', '--num_workers', type=int, default=None)
@click.option('--force', is_flag=True, default=False, help='Re-generate even if tag_detection.pkl exists')
def main(input_dir, camera_intrinsics, aruco_yaml, num_workers, force):
    input_dir = pathlib.Path(os.path.expanduser(input_dir))
    input_video_dirs = [x.parent for x in input_dir.glob('*/raw_video.mp4')]
    print(f'Found {len(input_video_dirs)} video dirs')

    assert os.path.isfile(camera_intrinsics)
    assert os.path.isfile(aruco_yaml)

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    script_path = pathlib.Path(__file__).parent.parent.joinpath('scripts', 'detect_aruco.py')

    failed = []

    with tqdm(total=len(input_video_dirs)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()

            def _run(cmd, name):
                r = subprocess.run(cmd, capture_output=True, text=True)
                return name, r.returncode, r.stdout, r.stderr

            for video_dir in input_video_dirs:
                video_dir = video_dir.absolute()
                video_path = video_dir.joinpath('raw_video.mp4')
                pkl_path = video_dir.joinpath('tag_detection.pkl')
                ts_csv = video_dir.joinpath('raw_video_timestamps.csv')

                if (not force) and pkl_path.is_file():
                    print(f"tag_detection.pkl already exists, skipping {video_dir.name}")
                    pbar.update(1)
                    continue

                if not ts_csv.is_file():
                    print(f"[WARN] missing raw_video_timestamps.csv, skipping {video_dir.name}")
                    pbar.update(1)
                    continue

                cmd = [
                    sys.executable, str(script_path),
                    '--input', str(video_path),
                    '--output', str(pkl_path),
                    '--intrinsics_json', str(camera_intrinsics),
                    '--aruco_yaml', str(aruco_yaml),
                    '--num_workers', '1',
                    '--timestamps_csv', str(ts_csv),
                    '--save-pts-time'
                ]

                if len(futures) >= num_workers:
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for fut in completed:
                        name, code, out, err = fut.result()
                        if code != 0:
                            failed.append((name, code, err))
                            print(f"[FAIL] {name} rc={code}\n{err}")
                        pbar.update(1)

                futures.add(executor.submit(_run, cmd, video_dir.name))

            completed, futures = concurrent.futures.wait(futures)
            for fut in completed:
                name, code, out, err = fut.result()
                if code != 0:
                    failed.append((name, code, err))
                    print(f"[FAIL] {name} rc={code}\n{err}")
                pbar.update(1)

    if failed:
        print("\n========== FAILED JOBS ==========")
        for name, code, err in failed[:20]:
            print(f"- {name} rc={code}: {err[:400]}")
        print("================================\n")
        raise SystemExit(f"{len(failed)} jobs failed.")
    else:
        print("Done! All jobs succeeded.")


if __name__ == "__main__":
    main()