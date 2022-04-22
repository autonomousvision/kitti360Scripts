#!/bin/bash

# This is NOT the real GT! In this demo, we compare the difference of Semantic SUMA and ORB-SLAM 
# to illustrate how the APE and RPE is evaluated on the KITTI-360 localication benchmark
gt_dir=./test_data/semantic_suma
reconstruction_dir=./test_data/orbslam

out_dir=./results/$1/output
mkdir -p $out_dir


rm $out_dir/*

test_ids=("0" "1" "2" "3")

for test_id in "${test_ids[@]}"
do

    if [ -f ${out_dir}/${test_id}_rpe.zip ]; then
    	rm ${out_dir}/${test_id}_rpe.zip
    fi
    if [ -f ${out_dir}/${test_id}_ape.zip ]; then
    	rm ${out_dir}/${test_id}_ape.zip
    fi
    
   evo_rpe kitti ${gt_dir}/test_poses_${test_id}.txt ${reconstruction_dir}/test_poses_${test_id}.txt \
    	-va \
	--delta 1.0 \
    	--delta_unit m \
        --plot_mode xy \
    	--save_plot ${out_dir}/${test_id}_rpe.pdf \
    	--save_results ${out_dir}/${test_id}_rpe.zip
    
   evo_ape kitti ${gt_dir}/test_poses_${test_id}.txt ${reconstruction_dir}/test_poses_${test_id}.txt \
    	-r trans_part \
        -va \
        --plot_mode xy \
    	--save_plot ${out_dir}/${test_id}_ape.pdf \
    	--save_results ${out_dir}/${test_id}_ape.zip

done
