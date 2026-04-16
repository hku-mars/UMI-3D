#!/usr/bin/env bash
set -e

source devel/setup.bash
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '^/opt/MVS/lib/64$' | paste -sd: -)
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '^/opt/MVS/lib/32$' | paste -sd: -)

roslaunch umi_3d_slam mapping_mid360_180_min.launch rviz:=false

