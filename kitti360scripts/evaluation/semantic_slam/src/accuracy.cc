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

#include "accuracy.h"
#include "commons.h"

#include <Eigen/StdVector>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/search/kdtree.h>

// Number of cells used for the spatial access structure in inclination and
// azimuth directions.
const int kCellCountInclination = 1024;
const int kCellCountAzimuth = 2 * kCellCountInclination;

const int kGridCount = 2;
const float kGridShifts[kGridCount][3] = {{0.f, 0.f, 0.f}, {0.5f, 0.5f, 0.5f}};

// Accuracy results for one voxel cell.
struct AccuracyCell {
  inline AccuracyCell() : accurate_count(0), inaccurate_count(0) {}

  // Number of accurate reconstruction points within this cell.
  size_t accurate_count;

  // Number of inaccurate reconstruction points within this cell.
  size_t inaccurate_count;
};

void ComputePointAccuracy(const PointCloudPtr& reconstruction,
			 // global indices of each reconstruction point
			 const std::vector<int> reconstruction_point_global_indices,
                         const PointCloudPtr& scan,
			 // free or occupied region obtained from offline Octomap
                         const PointCloudPtr& occupied_or_free_voxel_centers,
			 const float occupied_or_free_voxel_size,
                         // Sorted by increasing tolerance.
                         const std::vector<float>& sorted_tolerances,
			 std::vector<int>* first_accuracy_tolerance_index,
			 std::vector<bool>* inaccurate_classifications_exist,
                         // Indexed by: [tolerance_index][reconstruction_point_index].
    			 std::vector<std::vector<AccuracyResult>>* point_is_accurate){
  bool output_point_results = point_is_accurate != nullptr;
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
  pcl::search::KdTree<pcl::PointXYZ> scan_kdtree;
  // Get sorted results from radius search. True should be the default, but be
  // on the safe side for the case of changing defaults:
  scan_kdtree.setSortedResults(true);
  scan_kdtree.setInputCloud(scan);

  pcl::search::KdTree<pcl::PointXYZ> observed_kdtree;
  // Get sorted results from radius search. True should be the default, but be
  // on the safe side for the case of changing defaults:
  observed_kdtree.setSortedResults(true);
  observed_kdtree.setInputCloud(occupied_or_free_voxel_centers);

  const int kNN = 1;
  std::vector<int> knn_indices(kNN);
  std::vector<float> knn_squared_dists(kNN);

  const long long int reconstruction_point_size =
      static_cast<long long int>(reconstruction->size());

  // Loop over all reconstruction points.
  for (long long int reconstruction_point_index = 0; reconstruction_point_index < reconstruction_point_size;
       ++reconstruction_point_index) {
    long long int reconstruction_point_global_index = reconstruction_point_global_indices[reconstruction_point_index];
    // Skip if a point is evaluated as accurate for all thresholds in previous windows
    if (first_accuracy_tolerance_index->at(reconstruction_point_global_index) == 0){
            continue;
    }

    const pcl::PointXYZ& reconstruction_point = reconstruction->at(reconstruction_point_index);


    // Find the closest voxel center to check if the reconstruction point falls in the unobserved region
    search_point.getVector3fMap() = reconstruction_point.getVector3fMap();
    if (observed_kdtree.radiusSearch(search_point, occupied_or_free_voxel_size * 1.414f,
                                           knn_indices, knn_squared_dists,
                                           kNN) == 0) {
            if (output_point_results) {
                for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
                     ++tolerance_index) {
                    point_is_accurate->at(tolerance_index)[reconstruction_point_global_index] = AccuracyResult::kUnobserved;
		}
	    }
	    continue;
    }

    // Find the closest scan point to this reconstruction point, limited to the
    // maximum evaluation tolerance for efficiency.
    if (scan_kdtree.radiusSearch(search_point, maximum_tolerance,
                                           knn_indices, knn_squared_dists,
                                           kNN) > 0) {
      int smallest_accurate_tolerance_index = 0;
      {
        // Find the smallest tolerance for which it is still accurate.
        for (int tolerance_index = tolerances_count - 2; tolerance_index >= 0;
             --tolerance_index) {
          if (sorted_tolerances_squared[tolerance_index] <
              knn_squared_dists[0]) {
            // The reconstruction point is not accurate for the current tolerance index.
            smallest_accurate_tolerance_index = tolerance_index + 1;
            break;
          }
        }
      }

      // If the point was not traversed before or if a better accuracy result is obtained 
      if (first_accuracy_tolerance_index->at(reconstruction_point_global_index)<0 ||
          smallest_accurate_tolerance_index < first_accuracy_tolerance_index->at(reconstruction_point_global_index)){

          first_accuracy_tolerance_index->at(reconstruction_point_global_index) = smallest_accurate_tolerance_index;
	  if (smallest_accurate_tolerance_index>0){
		  inaccurate_classifications_exist->at(reconstruction_point_global_index)=true;
	  }

          // Output point results, if requested.
          if (output_point_results) {
            // The points is inaccurate for tolerances smaller than the smallest
            // accurate one.
            for (int tolerance_index = 0;
                 tolerance_index < smallest_accurate_tolerance_index;
                 ++tolerance_index) {
              point_is_accurate->at(tolerance_index)[reconstruction_point_global_index] = AccuracyResult::kInaccurate;
            }
            // The point is accurate for tolerances starting from the smallest
            // accurate one.
            for (size_t tolerance_index = smallest_accurate_tolerance_index;
                 tolerance_index < tolerances_count; ++tolerance_index) {
              point_is_accurate->at(tolerance_index)[reconstruction_point_global_index] = AccuracyResult::kAccurate;
            }
          }
      }
    } else if (first_accuracy_tolerance_index->at(reconstruction_point_global_index)<0) {
      if (output_point_results){
        // This reconstruction point is inaccurate for all tolerances.
        for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
             ++tolerance_index) {
          point_is_accurate->at(tolerance_index)[reconstruction_point_global_index] = AccuracyResult::kInaccurate;
        }
      }
      first_accuracy_tolerance_index->at(reconstruction_point_global_index) = tolerances_count;
      inaccurate_classifications_exist->at(reconstruction_point_global_index) = true;
    }
  }

}

void ComputeVoxelAccuracy(
    const PointCloud& reconstruction,
    const std::vector<int>& first_accurate_tolerance_indices,
    const std::vector<bool>& inaccurate_classifications_exist,
    float voxel_size_inv,
    // Sorted by increasing tolerance.
    const std::vector<float>& sorted_tolerances, 
    // Indexed by: [tolerance_index]. Range: [0, 1].
    std::vector<float>* results){

  //size_t scan_count = scans.size();
  size_t tolerances_count = sorted_tolerances.size();

  // Differently shifted voxel grids.
  // Indexed by: [map_index][CalcCellCoordinates(...)][tolerance_index].
  std::unordered_map<std::tuple<int, int, int>, std::vector<AccuracyCell>>
      cell_maps[kGridCount];

  // Loop over the reconstruction points.
  for (size_t point_index = 0, size = reconstruction.size(); point_index < size;
       ++point_index) {
    const pcl::PointXYZ& point = reconstruction.at(point_index);

    // Find the voxels for this reconstruction point.
    std::vector<AccuracyCell>* cell_vectors[kGridCount];
    for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
      std::vector<AccuracyCell>* cell_vector =
          &cell_maps[grid_index][CalcCellCoordinates(
              point, voxel_size_inv, kGridShifts[grid_index][0],
              kGridShifts[grid_index][1], kGridShifts[grid_index][2])];
      if (cell_vector->empty()) {
        cell_vector->resize(tolerances_count);
      }
      cell_vectors[grid_index] = cell_vector;
    }

    int aggregate_first_accurate_tolerance_index =
	    first_accurate_tolerance_indices[point_index];
    bool aggregate_inaccurate_classifications_exist = 
	    inaccurate_classifications_exist[point_index];
    if (aggregate_first_accurate_tolerance_index==-1){
	    continue;
    }

    // Aggregate accurate count.
    for (int tolerance_index = aggregate_first_accurate_tolerance_index;
         tolerance_index < static_cast<int>(tolerances_count);
         ++tolerance_index) {
      for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
        ++cell_vectors[grid_index]->at(tolerance_index).accurate_count;
      }
    }
    // Aggregate inaccurate count or unobserved count.
    if (aggregate_inaccurate_classifications_exist) {
      for (int tolerance_index = aggregate_first_accurate_tolerance_index - 1;
           tolerance_index >= 0; --tolerance_index) {
        for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
          ++cell_vectors[grid_index]->at(tolerance_index).inaccurate_count;
        }
      }
    }
  }

  // Average results over all cells and fill the results vector.
  std::vector<double> accuracy_sum(tolerances_count, 0.0);
  std::vector<size_t> valid_cell_count(tolerances_count, 0);
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    for (auto it = cell_maps[grid_index].cbegin(),
              end = cell_maps[grid_index].cend();
         it != end; ++it) {
      const std::vector<AccuracyCell>& cell_vector = it->second;
      for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
           ++tolerance_index) {
        const AccuracyCell& cell = cell_vector[tolerance_index];
        size_t valid_point_count = cell.accurate_count + cell.inaccurate_count;
        if (valid_point_count > 0) {
          accuracy_sum[tolerance_index] +=
              cell.accurate_count / (1.0f * valid_point_count);
          ++valid_cell_count[tolerance_index];
        }
      }
    }
  }

  results->resize(tolerances_count);
  for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
       ++tolerance_index) {
    float* accuracy = &results->at(tolerance_index);
    if (valid_cell_count[tolerance_index] == 0) {
      *accuracy = 0;
    } else {
      *accuracy =
          accuracy_sum[tolerance_index] / valid_cell_count[tolerance_index];
    }
  }
}


void WriteAccuracyVisualization(
    const std::string& base_path, const PointCloud& reconstruction,
    // Sorted by increasing tolerance.
    const std::vector<float>& sorted_tolerances,
    // Indexed by: [tolerance_index][point_index].
    const std::vector<std::vector<AccuracyResult>>& point_is_accurate,
    const int window_index) {
  pcl::PointCloud<pcl::PointXYZRGB> accuracy_visualization;
  accuracy_visualization.resize(reconstruction.size());

  // Set visualization point positions (once for all tolerances).
  for (size_t i = 0; i < reconstruction.size(); ++i) {
    accuracy_visualization.at(i).getVector3fMap() =
        reconstruction.at(i).getVector3fMap();
  }

  // Loop over all tolerances, set visualization point colors accordingly and
  // save the point clouds.
  for (size_t tolerance_index = 0; tolerance_index < sorted_tolerances.size();
       ++tolerance_index) {
    const std::vector<AccuracyResult>& point_is_accurate_for_tolerance =
        point_is_accurate[tolerance_index];

    for (size_t point_index = 0; point_index < accuracy_visualization.size();
         ++point_index) {
      pcl::PointXYZRGB* point = &accuracy_visualization.at(point_index);
      if (point_is_accurate_for_tolerance[point_index] ==
          AccuracyResult::kAccurate) {
        // Green: accurate points.
        point->r = 0;
        point->g = 255;
        point->b = 0;
      } else if (point_is_accurate_for_tolerance[point_index] ==
                 AccuracyResult::kInaccurate) {
        // Red: inaccurate points.
        point->r = 255;
        point->g = 0;
        point->b = 0;
      } else if (point_is_accurate_for_tolerance[point_index] ==
                 AccuracyResult::kUnobserved) {
        // Blue: unobserved points.
        point->r = 0;
        point->g = 0;
        point->b = 255;
      }
    }

    std::ostringstream file_path;
    file_path << base_path << window_index <<".tolerance_"
              << sorted_tolerances[tolerance_index] << ".ply";
    pcl::io::savePLYFileBinary(file_path.str(), accuracy_visualization);
    //WriteBinaryFile(file_path.str(), accuracy_visualization); 
  }
}

void ComputeAverageAccuracy(
    const PointCloud& reconstruction,
    const std::vector<int>& first_accurate_tolerance_indices,
    const std::vector<bool>& inaccurate_classifications_exist,
    // Sorted by increasing tolerance.
    const std::vector<float>& sorted_tolerances, 
    // Indexed by: [tolerance_index]. Range: [0, 1].
    std::vector<float>* results){

  //size_t scan_count = scans.size();
  size_t tolerances_count = sorted_tolerances.size();

  // Loop over the reconstruction points.
  std::vector<size_t> accurate_sum(tolerances_count, 0);
  std::vector<size_t> inaccurate_sum(tolerances_count, 0);
  for (size_t point_index = 0, size = reconstruction.size(); point_index < size;
       ++point_index) {
    const pcl::PointXYZ& point = reconstruction.at(point_index);

    int aggregate_first_accurate_tolerance_index =
	    first_accurate_tolerance_indices[point_index];
    bool aggregate_inaccurate_classifications_exist = 
	    inaccurate_classifications_exist[point_index];
    if (aggregate_first_accurate_tolerance_index==-1){
	    continue;
    }

    // Aggregate accurate count.
    for (int tolerance_index = aggregate_first_accurate_tolerance_index;
         tolerance_index < static_cast<int>(tolerances_count);
         ++tolerance_index) {
        accurate_sum[tolerance_index] ++;
    }
    // Aggregate inaccurate count or unobserved count.
    if (aggregate_inaccurate_classifications_exist) {
      for (int tolerance_index = aggregate_first_accurate_tolerance_index - 1;
           tolerance_index >= 0; --tolerance_index) {
          inaccurate_sum[tolerance_index] ++;
      }
    }
  }

  // Average results over all cells and fill the results vector.
  results->resize(tolerances_count);
  std::vector<size_t> valid_cell_count(tolerances_count, 0);
  for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
       ++tolerance_index) {
    float* accuracy = &results->at(tolerance_index);
    valid_cell_count[tolerance_index] = accurate_sum[tolerance_index] + inaccurate_sum[tolerance_index];
    if (valid_cell_count[tolerance_index] == 0) {
      *accuracy = 0;
    } else {
      *accuracy =
          (float)(accurate_sum[tolerance_index]) / (float)(valid_cell_count[tolerance_index]);
    }
  }
}

