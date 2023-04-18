// The evaluation code is built and modified based on the following repo: 
// https://github.com/ETH3D/multi-view-evaluation
// following copyright and permission notice:  
//
// Copyright 2017 Thomas Schöps, Johannes L. Schönberger
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

#include "completeness.h"
#include "commons.h"

#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/search/kdtree.h>

const int kGridCount = 2;
const float kGridShifts[kGridCount][3] = {{0.f, 0.f, 0.f}, {0.5f, 0.5f, 0.5f}};

// Completeness results for one voxel cell.
struct CompletenessCell {
  inline CompletenessCell() : point_count(0) {}

  // Number of scan points within this cell.
  size_t point_count;

  // Number of complete scan points (smaller or equal to point_count), for each
  // tolerance value.
  // Indexed by: [tolerance_index].
  std::vector<size_t> complete_count;
};

void ComputePointCompleteness(const PointCloudPtr& scan,
			 // global indices of each scan point
			 const std::vector<int> scan_point_global_indices,
                         const PointCloudPtr& reconstruction,
			 // global indices of each reconstruction point
			 const std::vector<int> reconstruction_point_global_indices,
                         // Sorted by increasing tolerance.
                         const std::vector<float>& sorted_tolerances,
			 std::vector<int>* first_completeness_tolerance_index,
                         // Indexed by: [tolerance_index][scan_point_index].
                         std::vector<std::vector<float>>* scan_nn_point_distances,
                         // Indexed by: [tolerance_index][scan_point_index].
                         std::vector<std::vector<int>>* scan_nn_point_indices,
                         // Indexed by: [tolerance_index][scan_point_index].
                         std::vector<std::vector<bool>>* point_is_complete) {
  bool output_point_results = point_is_complete != nullptr;
  size_t tolerances_count = sorted_tolerances.size();
  float maximum_tolerance = sorted_tolerances.back();

  // Compute squared tolerances.
  std::vector<float> sorted_tolerances_squared(tolerances_count);
  for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
       ++tolerance_index) {
    sorted_tolerances_squared[tolerance_index] =
        sorted_tolerances[tolerance_index] * sorted_tolerances[tolerance_index];
  }

  // Special case for empty reconstructions since the KdTree construction
  // crashes for them.
  if (reconstruction->size() == 0) {
    return;
  }

  pcl::PointXYZ search_point;
  pcl::search::KdTree<pcl::PointXYZ> reconstruction_kdtree;
  // Get sorted results from radius search. True should be the default, but be
  // on the safe side for the case of changing defaults:
  reconstruction_kdtree.setSortedResults(true);
  reconstruction_kdtree.setInputCloud(reconstruction);

  const int kNN = 1;
  std::vector<int> knn_indices(kNN);
  std::vector<float> knn_squared_dists(kNN);
  int nn_index;

  const long long int scan_point_size =
      static_cast<long long int>(scan->size());

  // Loop over all scan points.
  for (long long int scan_point_index = 0; scan_point_index < scan_point_size;
       ++scan_point_index) {
    long long int scan_point_global_index = scan_point_global_indices[scan_point_index];
    // Skip if a point is evaluated as complete for all thresholds in previous windows
    //if (first_completeness_tolerance_index->at(scan_point_global_index) == 0){
    //        continue;
    //}

    const pcl::PointXYZ& scan_point = scan->at(scan_point_index);

    // Find the closest reconstruction point to this scan point, limited to the
    // maximum evaluation tolerance for efficiency.
    search_point.getVector3fMap() = scan_point.getVector3fMap();


    if (reconstruction_kdtree.radiusSearch(search_point, maximum_tolerance,
                                           knn_indices, knn_squared_dists,
                                           kNN) > 0) {
      int smallest_complete_tolerance_index = 0;
      nn_index = reconstruction_point_global_indices[knn_indices[0]];
      {

        // Next, find the smallest tolerance for which it is still complete.
        for (int tolerance_index = tolerances_count - 2; tolerance_index >= 0;
             --tolerance_index) {
          if (sorted_tolerances_squared[tolerance_index] <
              knn_squared_dists[0]) {
            // The scan point is not completed for the current tolerance index.
            smallest_complete_tolerance_index = tolerance_index + 1;
            break;
          }
        }
      }

      // If the point is unobserved or if a better completeness result is obtained 
      // or if a closer nearest neighbor is found
      // We keep the closest nearest neighbor for evaluating semantic prediction
      if (first_completeness_tolerance_index->at(scan_point_global_index)<0 ||
          smallest_complete_tolerance_index < first_completeness_tolerance_index->at(scan_point_global_index) ||
          knn_squared_dists[0] < scan_nn_point_distances->at(smallest_complete_tolerance_index)[scan_point_global_index]){

          first_completeness_tolerance_index->at(scan_point_global_index) = smallest_complete_tolerance_index;
          // The point is complete for tolerances starting from the smallest
          // complete one.
          for (size_t tolerance_index = smallest_complete_tolerance_index;
               tolerance_index < tolerances_count; ++tolerance_index){
            if(knn_squared_dists[0] < scan_nn_point_distances->at(tolerance_index)[scan_point_global_index]){
	        scan_nn_point_indices->at(tolerance_index)[scan_point_global_index] = nn_index;
                scan_nn_point_distances->at(tolerance_index)[scan_point_global_index] = knn_squared_dists[0];
            }
          }

          // Output point results, if requested.
          if (output_point_results) {
            // The points is incomplete for tolerances smaller than the smallest
            // complete one.
            for (int tolerance_index = 0;
                 tolerance_index < smallest_complete_tolerance_index;
                 ++tolerance_index) {
              point_is_complete->at(tolerance_index)[scan_point_global_index] = false;
            }
            // The point is complete for tolerances starting from the smallest
            // complete one.
            for (size_t tolerance_index = smallest_complete_tolerance_index;
                 tolerance_index < tolerances_count; ++tolerance_index) {
              point_is_complete->at(tolerance_index)[scan_point_global_index] = true;
	      //scan_nn_point_indices->at(tolerance_index)[scan_point_global_index] = nn_index;
            }
          }
      }
    } else if (first_completeness_tolerance_index->at(scan_point_global_index)<0) {
      if (output_point_results){
        // This scan point is incomplete for all tolerances.
        for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
             ++tolerance_index) {
          point_is_complete->at(tolerance_index)[scan_point_global_index] = false;
        }
      }
      first_completeness_tolerance_index->at(scan_point_global_index) = tolerances_count+1;
    }
  }

}

void ComputeVoxelCompleteness(const PointCloudPtr& scan,
                         float voxel_size_inv,
                         // Sorted by increasing tolerance.
                         const std::vector<float>& sorted_tolerances,
                         // Indexed by: [scan_point_global_index].
			 const std::vector<int>& first_completeness_tolerance_index,
                         // Indexed by: [tolerance_index]. Range: [0, 1].
                         std::vector<float>* results){
  size_t tolerances_count = sorted_tolerances.size();
  float maximum_tolerance = sorted_tolerances.back();

  // Differently shifted voxel grids.
  // Indexed by: [map_index][CalcCellCoordinates(...)].
  std::unordered_map<std::tuple<int, int, int>, CompletenessCell>
      cell_maps[kGridCount];

  const long long int scan_point_size =
      static_cast<long long int>(scan->size());

  // Loop over all scan points.
  for (long long int scan_point_index = 0; scan_point_index < scan_point_size;
       ++scan_point_index) {
    const pcl::PointXYZ& scan_point = scan->at(scan_point_index);

    // Find the voxels for this scan point and increase their point count.
    CompletenessCell* cells[kGridCount];
    for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
      CompletenessCell* cell = &cell_maps[grid_index][CalcCellCoordinates(
          scan_point, voxel_size_inv, kGridShifts[grid_index][0],
          kGridShifts[grid_index][1], kGridShifts[grid_index][2])];
      ++cell->point_count;
      if (cell->complete_count.empty()) {
        cell->complete_count.resize(tolerances_count, 0);
      }
      cells[grid_index] = cell;
    }

    // Find the closest reconstruction point to this scan point, limited to the
    // maximum evaluation tolerance for efficiency.
    if (first_completeness_tolerance_index[scan_point_index]<tolerances_count) {
      {
        // Next, find the smallest tolerance for which it is still complete.
        for (int tolerance_index = tolerances_count - 1; tolerance_index >= first_completeness_tolerance_index[scan_point_index];
             --tolerance_index) {
          for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
            cells[grid_index]->complete_count[tolerance_index] += 1;
          }
        }
      }
    }
  }

  // Average results over all cells and fill the results vector.
  std::vector<double> completeness_sum(tolerances_count, 0.0);
  size_t cell_count = 0;
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    cell_count += cell_maps[grid_index].size();

    for (auto it = cell_maps[grid_index].cbegin(),
              end = cell_maps[grid_index].cend();
         it != end; ++it) {
      const CompletenessCell& cell = it->second;
      for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
           ++tolerance_index) {
        completeness_sum[tolerance_index] +=
            cell.complete_count[tolerance_index] / (1.0 * cell.point_count);
      }
    }
  }

  results->resize(tolerances_count);
  for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
       ++tolerance_index) {
    results->at(tolerance_index) =
        completeness_sum[tolerance_index] / cell_count;
  }

}

void WriteCompletenessVisualization(
    const std::string& base_path,
    const std::map<int, Eigen::Matrix4f> &scan_infos,
    //const MeshLabMeshInfoVector& scan_infos,
    //const std::map<int, PointCloudPtr>& scans,
    const PointCloudPtr& scan,
    // Sorted by increasing tolerance.
    const std::vector<float>& sorted_tolerances,
    // Indexed by: [tolerance_index][scan_point_index].
    const std::vector<std::vector<bool>>& point_is_complete,
    const int window_index) {
  pcl::PointCloud<pcl::PointXYZRGB> completeness_visualization;
  completeness_visualization.resize(point_is_complete[0].size());

  // Set visualization point positions (once for all tolerances).
  copyPointCloud(*scan, completeness_visualization);

  // Loop over all tolerances, set visualization point colors accordingly and
  // save the point clouds.
  for (size_t tolerance_index = 0; tolerance_index < sorted_tolerances.size();
       ++tolerance_index) {
    const std::vector<bool>& point_is_complete_for_tolerance =
        point_is_complete[tolerance_index];

    for (size_t scan_point_index = 0;
         scan_point_index < completeness_visualization.size();
         ++scan_point_index) {
      pcl::PointXYZRGB* point =
          &completeness_visualization.at(scan_point_index);
      if (point_is_complete_for_tolerance[scan_point_index]) {
        // Green: complete points.
        point->r = 0;
        point->g = 255;
        point->b = 0;
      } else {
        // Red: incomplete points.
        point->r = 255;
        point->g = 0;
        point->b = 0;
      }
    }

    std::ostringstream file_path;
    file_path << base_path << window_index << ".tolerance_"
              << sorted_tolerances[tolerance_index] << ".ply";
    pcl::io::savePLYFileBinary(file_path.str(), completeness_visualization);
    //WriteBinaryFile(file_path.str(), completeness_visualization); 
  }
}

void ComputeAverageCompleteness(const PointCloudPtr& scan,
                         // Sorted by increasing tolerance.
                         const std::vector<float>& sorted_tolerances,
                         // Indexed by: [scan_point_global_index].
			 const std::vector<int>& first_completeness_tolerance_index,
                         // Indexed by: [tolerance_index]. Range: [0, 1].
                         std::vector<float>* results){
  size_t tolerances_count = sorted_tolerances.size();
  float maximum_tolerance = sorted_tolerances.back();

  // Differently shifted voxel grids.
  // Indexed by: [map_index][CalcCellCoordinates(...)].
  std::unordered_map<std::tuple<int, int, int>, CompletenessCell>
      cell_maps[kGridCount];

  const long long int scan_point_size =
      static_cast<long long int>(scan->size());

  // Loop over all scan points.
  std::vector<size_t> complete_sum(tolerances_count, 0);
  for (long long int scan_point_index = 0; scan_point_index < scan_point_size;
       ++scan_point_index) {
    const pcl::PointXYZ& scan_point = scan->at(scan_point_index);

    // Find the closest reconstruction point to this scan point, limited to the
    // maximum evaluation tolerance for efficiency.
    if (first_completeness_tolerance_index[scan_point_index]<tolerances_count) {
      {
        // Next, find the smallest tolerance for which it is still complete.
        for (int tolerance_index = tolerances_count - 1; tolerance_index >= first_completeness_tolerance_index[scan_point_index];
             --tolerance_index) {
            complete_sum[tolerance_index] ++;
        }
      }
    }
  }

  // Average results over all points and fill the results vector.
  results->resize(tolerances_count);
  for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
       ++tolerance_index) {
    results->at(tolerance_index) =
        (float) (complete_sum[tolerance_index]) / (float)scan_point_size;
  }

}
