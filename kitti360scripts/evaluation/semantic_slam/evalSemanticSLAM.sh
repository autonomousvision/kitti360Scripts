#!/bin/bash

reconstruction_dir=RECONSTRUCTION_DIR
gt_dir=GT_DIR
build_dir=./build

test_ids=("0" "1" "2" "3")

for test_id in "${test_ids[@]}"
do
    echo "==========  evaluating test sequence $test_id ========="
    output_dir=./${reconstruction_dir}/test_${test_id}
    mkdir -p $output_dir
    
    # evaluate completeness, accuracy and semantic at the threshold of 0.1m
    $build_dir/KITTI360SemanticSlamEvaluation --tolerances 0.1 \
    	--reconstruction_ply_path ${reconstruction_dir}/test_${test_id}.ply \
    	--reconstruction_pose_path ${reconstruction_dir}/test_poses_${test_id}.txt \
    	--reconstruction_semantic_path ${reconstruction_dir}/test_semantic_${test_id}.txt \
    	--ground_truth_data_path ${gt_dir}/gt_${test_id}.ply \
    	--ground_truth_pose_path ${gt_dir}/gt_poses_${test_id}.txt \
    	--ground_truth_observed_region_path ${gt_dir}/gt_observed_voxels_${test_id}.ply \
    	--ground_truth_semantic_path ${gt_dir}/gt_semantic_${test_id}.txt \
    	--ground_truth_conf_path ${gt_dir}/gt_confidence_binary_${test_id}.bin \
    	--completeness_cloud_output_path ${output_dir}/output_completeness \
    	--accuracy_cloud_output_path ${output_dir}/output_accuracy \
    	--result_output_path ${output_dir}/
        #--voxel_size 0.075

    test_id=$(( test_id + 1 ))
done

exit
