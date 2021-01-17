#!/bin/bash

# This script accumulates velodyne/sick point clouds
# Example usage:
#   ./run_accumulation.sh $KITTI360_DATASET $OUTPUT_DIR 2013_05_28_drive_0003_sync 2 282


set -e

# arguments
root_dir=$1
output_dir=$2
sequence=$3
first_frame=$4
last_frame=$5
data_source=$6 # 0: sick only, 1: velodyne only, 2: velodyne + sick

#parameters
#----------------------#
travel_padding=20 
verbose=1
space=0.005
#----------------------#

echo ===================================================
echo "Processing $sequence: [$first_frame,$last_frame]"
echo ===================================================

# point cloud accumulation (generate a folder called data_raw)
echo ./../accumuLaser/build/point_accumulator ${root_dir} ${sequence} ${output_dir} ${first_frame} ${last_frame} ${travel_padding} ${data_source} ${space} ${verbose}
./../accumuLaser/build/point_accumulator ${root_dir} ${sequence} ${output_dir} ${first_frame} ${last_frame} ${travel_padding} ${data_source} ${space} ${verbose}

# done

