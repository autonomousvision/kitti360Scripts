// The evaluation code is built and modified based on the following repo: 
// https://github.com/ETH3D/multi-view-evaluation
// following copyright and permission notice:  
//
// Copyright 2017 Thomas Schöps
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <string>
#include <vector>
#include <map>

#include "util.h"

enum class AccuracyResult : uint8_t {
  kUnobserved = 0,
  kAccurate = 1,
  kInaccurate = 2
};

// Computes the completeness of a partial point cloud, save results to a
// global output
void ComputePointAccuracy(const PointCloudPtr& reconstruction,
    // global indices of each scan point
    const std::vector<int> reconstruction_point_global_indices,
    const PointCloudPtr& scan,
    // free or occupied region obtained from offline Octomap
    const PointCloudPtr& occupied_or_free_voxel_centers,
    const float occupied_or_free_voxel_size,
    // Sorted by increasing tolerance.
    const std::vector<float>& sorted_tolerances,
    // Indexed by: [scan_point_global_index].
    std::vector<int>* first_accuracy_tolerance_index,
    std::vector<bool>* inaccurate_classifications_exist,
    // Indexed by: [tolerance_index][scan_point_index].
    std::vector<std::vector<AccuracyResult>>* point_is_accurate);

void ComputeVoxelAccuracy(
    const PointCloud& reconstruction,
    const std::vector<int>& first_accurate_tolerance_indices,
    const std::vector<bool>& inaccurate_classifications_exist,
    float voxel_size_inv,
    // Sorted by increasing tolerance.
    const std::vector<float>& sorted_tolerances, 
    // Indexed by: [tolerance_index]. Range: [0, 1].
    std::vector<float>* results);

void ComputeAverageAccuracy(
    const PointCloud& reconstruction,
    const std::vector<int>& first_accurate_tolerance_indices,
    const std::vector<bool>& inaccurate_classifications_exist,
    // Sorted by increasing tolerance.
    const std::vector<float>& sorted_tolerances, 
    // Indexed by: [tolerance_index]. Range: [0, 1].
    std::vector<float>* results);

void WriteAccuracyVisualization(
    const std::string& base_path,
    const PointCloud& reconstruction,
    // Sorted by increasing tolerance.
    const std::vector<float>& sorted_tolerances,
    // Indexed by: [tolerance_index][point_index].
    const std::vector<std::vector<AccuracyResult>>& point_is_accurate,
    const int window_index=-1);
