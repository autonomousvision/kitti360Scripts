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

#include <Eigen/Core>
#include <boost/filesystem.hpp>
#include <pcl/console/parse.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>


#include "accuracy.h"
#include "completeness.h"
#include "semantic.h"
#include "util.h"
#include "commons.h"
#include "pose.h"

const float kDegToRadFactor = M_PI / 180.0;
const float occupied_or_free_voxel_size = 0.5;
const int semantic_class_count=19;
const bool debug = false;

// Return codes of the program:
// 0: Success.
// 1: System failure (e.g., due to wrong parameters given).
// 2: Reconstruction file input failure (PLY file cannot be found or read).
enum class ReturnCodes {
  kSuccess = 0,
  kSystemFailure = 1,
  kReconstructionFileInputFailure = 2
};

int main(int argc, char** argv) {
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

  // Parse arguments.
  std::string reconstruction_ply_path;
  pcl::console::parse_argument(argc, argv, "--reconstruction_ply_path",
                               reconstruction_ply_path);
  std::string reconstruction_pose_path;
  pcl::console::parse_argument(argc, argv, "--reconstruction_pose_path",
                               reconstruction_pose_path);
  std::string reconstruction_semantic_path;
  pcl::console::parse_argument(argc, argv, "--reconstruction_semantic_path",
                               reconstruction_semantic_path);
  std::string ground_truth_pose_path;
  pcl::console::parse_argument(argc, argv, "--ground_truth_pose_path",
                               ground_truth_pose_path);
  std::string ground_truth_data_path;
  pcl::console::parse_argument(argc, argv, "--ground_truth_data_path",
                               ground_truth_data_path);
  std::string ground_truth_conf_path;
  pcl::console::parse_argument(argc, argv, "--ground_truth_conf_path",
                               ground_truth_conf_path);
  std::string ground_truth_observed_region_path;
  pcl::console::parse_argument(argc, argv, "--ground_truth_observed_region_path",
                               ground_truth_observed_region_path);
  std::string ground_truth_semantic_path;
  pcl::console::parse_argument(argc, argv, "--ground_truth_semantic_path",
                               ground_truth_semantic_path);
  std::vector<float> tolerances;
  pcl::console::parse_x_arguments(argc, argv, "--tolerances", tolerances);
  float voxel_size = 0.01f;
  pcl::console::parse_argument(argc, argv, "--voxel_size", voxel_size);
  float beam_start_radius_meters = 0.5 * 0.00225;
  pcl::console::parse_argument(argc, argv, "--beam_start_radius_meters",
                               beam_start_radius_meters);
  float beam_divergence_halfangle_deg = 0.011;
  pcl::console::parse_argument(argc, argv, "--beam_divergence_halfangle_deg",
                               beam_divergence_halfangle_deg);
  std::string completeness_cloud_output_path;
  pcl::console::parse_argument(argc, argv, "--completeness_cloud_output_path",
                               completeness_cloud_output_path);
  std::string accuracy_cloud_output_path;
  pcl::console::parse_argument(argc, argv, "--accuracy_cloud_output_path",
                               accuracy_cloud_output_path);
  std::string result_output_path;
  pcl::console::parse_argument(argc, argv, "--result_output_path",
                               result_output_path);

  // Validate arguments.
  std::stringstream errors;
  if (tolerances.empty()) {
    errors << "The --tolerances parameter must be given as a list of"
           << " non-negative evaluation tolerance values, separated by"
           << " commas." << std::endl;
  }
  if (reconstruction_ply_path.empty()) {
    errors << "The --reconstruction_ply_path parameter must be given."
           << std::endl;
  }
  if (ground_truth_data_path.empty()) {
    errors << "The --ground_truth_data_path parameter must be given."
           << std::endl;
  }
  if (voxel_size <= 0.f) {
    errors << "The voxel size must be positive." << std::endl;
  }

  if (!errors.str().empty()) {
    std::cerr << "Usage example: " << argv[0]
              << " --tolerances 0.1,0.2 --reconstruction_ply_path "
                 "path/to/reconstruction.ply --reconstruction_pose_path "
		 "path/to/reconstruction-pose.txt --ground_truth_data_path "
                 "path/to/ground-truth.ply --ground_truth_ts_path "
                 "path/to/ground-truth-timestamps.txt --ground_truth_pose_path "
		 "path/to/ground-truth-pose.txt"
              << std::endl << std::endl;
    std::cerr << errors.str() << std::endl;
    return static_cast<int>(ReturnCodes::kSystemFailure);
  }

  // Process arguments.
  std::sort(tolerances.begin(), tolerances.end());
  float tan_beam_divergence_halfangle_rad =
      tan(kDegToRadFactor * beam_divergence_halfangle_deg);
  float voxel_size_inv = 1.0 / voxel_size;

  // Load ground truth point cloud poses
  std::map<int, Eigen::Matrix4f> ground_truth_poses;
  if (!LoadPoses(ground_truth_pose_path, ground_truth_poses)) {
    std::cerr << "Cannot read scan poses from " << ground_truth_pose_path
              << std::endl;
    return static_cast<int>(ReturnCodes::kSystemFailure);
  }

  // Load the reconstruction point cloud.
  std::cout << "Loading reconstruction: " << reconstruction_ply_path
            << std::endl;
  PointCloudPtr reconstruction(new PointCloud());
  if (pcl::io::loadPLYFile(reconstruction_ply_path, *reconstruction) < 0) {
    std::cerr << "Cannot read reconstruction file." << std::endl;
    return static_cast<int>(ReturnCodes::kReconstructionFileInputFailure);
  }
  std::cout << "Reconstruction point cloud size: " << reconstruction->points.size() << std::endl;

  // Load reconstruction point cloud poses
  std::map<int, Eigen::Matrix4f> reconstruction_poses;
  if (!LoadPoses(reconstruction_pose_path, reconstruction_poses)) {
    std::cerr << "Cannot read reconstruction poses from " << reconstruction_pose_path
              << std::endl;
    return static_cast<int>(ReturnCodes::kSystemFailure);
  }

  // Sanity check of the poses
  if (ground_truth_poses.size() != reconstruction_poses.size()){
    std::cerr << "Wrong length of reconstruction poses  " << reconstruction_pose_path
              << std::endl;
    return static_cast<int>(ReturnCodes::kSystemFailure);
  }

  // Load observed voxel centers
  PointCloudPtr occupied_or_free_voxel_centers(new PointCloud());
  std::cout << "Loading observed voxel centers: " << ground_truth_observed_region_path 
            << std::endl;
  LoadVoxelCenters(ground_truth_observed_region_path, occupied_or_free_voxel_centers);
  std::cout << "Loaded observed voxel centers: " << occupied_or_free_voxel_centers->points.size() << std::endl;

  // Load the ground truth scan point clouds.
  std::cout << "Initializing local scans " << std::endl;
  std::map<int, PointCloudPtr> scans;
  std::map<int, Eigen::Matrix4f>::iterator it;
  std::vector<int> ground_truth_pose_timestamps;
  for (it=ground_truth_poses.begin(); it!=ground_truth_poses.end(); it++){
      scans.insert(pair<int, PointCloudPtr> (it->first, PointCloudPtr(new PointCloud())));
      ground_truth_pose_timestamps.push_back(it->first);
  }
  PointCloudPtr scan(new PointCloud());
  if (pcl::io::loadPLYFile(ground_truth_data_path, *scan) < 0) {
    std::cerr << "Cannot read ground truth file." << std::endl;
    return static_cast<int>(ReturnCodes::kReconstructionFileInputFailure);
  }
  std::cout << "Ground truth scan size: " << scan->points.size() << std::endl;

  // Load semantic labels of ground truth scan point clouds
  bool evaluate_semantic = !reconstruction_semantic_path.empty();
  std::vector<int> reconstruction_semantic, scan_semantic;
  std::vector<float> scan_confidence;
  if (evaluate_semantic){
	  LoadSemanticLabel(reconstruction_semantic_path, reconstruction_semantic);
	  LoadSemanticLabel(ground_truth_semantic_path, scan_semantic);
	  LoadConfidenceBinary(ground_truth_conf_path, scan_confidence);
  	  std::cout << "Reconstruction semantic size: " << reconstruction_semantic.size() << std::endl;
  	  std::cout << "Ground truth semantic size: " << scan_semantic.size() << std::endl;
  	  std::cout << "Ground truth confidence size: " << scan_confidence.size() << std::endl;
  }

  // Initialize output for completeness
  // Indexed by: [tolerance_index].
  std::vector<float> completeness_results;
  // Indexed by: [tolerance_index][scan_point_index].
  std::vector<std::vector<bool>> point_is_complete(tolerances.size());
  // Indexed by: [tolerance_index][scan_point_index].
  std::vector<std::vector<int>> scan_nn_point_indices(tolerances.size());
  // Indexed by: [tolerance_index][scan_point_index].
  std::vector<std::vector<float>> scan_nn_point_distances(tolerances.size());
  for (size_t tolerance_index = 0; tolerance_index < tolerances.size();
       ++tolerance_index) {
    point_is_complete[tolerance_index].resize(scan->points.size(), false);
    scan_nn_point_indices[tolerance_index].resize(scan->points.size(), -1);
    scan_nn_point_distances[tolerance_index].resize(scan->points.size(), 1000.0);
  }
  std::vector<int> first_completeness_tolerance_index(scan->points.size(), -1);
  bool output_point_completeness = !completeness_cloud_output_path.empty();

  // Initialize output for accuracy
  std::vector<int> first_accuracy_tolerance_indices(reconstruction->points.size(), -1);
  std::vector<bool> inaccurate_classifications_exist(reconstruction->points.size(), false);
  // Indexed by: [tolerance_index].
  std::vector<float> accuracy_results;
  // Indexed by: [tolerance_index][scan_point_index].
  std::vector<std::vector<AccuracyResult>> point_is_accurate(tolerances.size());
  for (size_t tolerance_index = 0; tolerance_index < tolerances.size();
       ++tolerance_index) {
    point_is_accurate[tolerance_index].resize(reconstruction->points.size(), AccuracyResult::kUnobserved);
  }
  bool output_point_accuracy = !accuracy_cloud_output_path.empty();

  // Initialize output for semantic
  std::vector<float> semantic_results(tolerances.size());

  // Loop over ground truth timestamps to evaluate completeness and accuracy for local regions
  // Align poses within local windows to avoid accumulated pose errors affecting the evaluation
  int interval=50; // evaluate at every 10 frames
  int window_half_size = 25;
  for (int window_index=interval/2; window_index<ground_truth_pose_timestamps.size()-interval*2; window_index+=interval){
      std::cout << "Processing " << window_index << "/" << ground_truth_pose_timestamps.size()-interval*2 << endl;
      // Get poses in a small window for both the ground truth and the reconstruction
      int current_timestamp = ground_truth_pose_timestamps[window_index];
      std::vector<Eigen::Matrix4f> ground_truth_pose_window;
      //std::map<int, Eigen::Matrix4f> ground_truth_pose_window_map;
      std::vector<Eigen::Matrix4f> reconstruction_pose_window;
      //pcl::PointXYZ window_center;
      for (int i=-window_half_size; i<window_half_size; i++){
          int timestamp_i = ground_truth_pose_timestamps[window_index+i];
	  // TODO: make sure that timestamps of the gt and the estimation match
	  ground_truth_pose_window.push_back(ground_truth_poses.at(timestamp_i));
	  //ground_truth_pose_window_map.insert(pair<int, Eigen::Matrix4f>(timestamp_i, ground_truth_poses.at(timestamp_i)));
	  reconstruction_pose_window.push_back(reconstruction_poses.at(timestamp_i));
	  //window_center.x += ground_truth_poses.at(timestamp_i).coeff(0,3);
	  //window_center.y += ground_truth_poses.at(timestamp_i).coeff(1,3);
	  //window_center.z += ground_truth_poses.at(timestamp_i).coeff(2,3);
      }
      //window_center.x /= (window_half_size*2);
      //window_center.y /= (window_half_size*2);
      //window_center.z /= (window_half_size*2);

      // Align ground truth poses and predicted poses and transform the reconstruction
      std::vector<Eigen::Matrix4f> aligned_reconstruction_pose_window;
      Eigen::Matrix4f alignment_transform;
      AlignTrajectory(ground_truth_pose_window, reconstruction_pose_window, 
		      aligned_reconstruction_pose_window, alignment_transform);

      // Get point clouds in a small window for both the ground truth and the reconstruction 
      PointCloudPtr reconstruction_window_unaligned(new PointCloud());
      PointCloudPtr scan_window(new PointCloud());
      std::vector<int> reconstruction_window_point_indices;
      std::vector<int> scan_window_point_indices;
      GetPointsInRange(reconstruction, reconstruction_pose_window, reconstruction_window_unaligned,
		       reconstruction_window_point_indices);
      GetPointsInRange(scan, ground_truth_pose_window, scan_window, scan_window_point_indices);

      if (scan_window->points.size()==0 || reconstruction_window_unaligned->points.size()==0){
	      continue;
      }

      //PointCloudPtr reconstruction_window_aligned_by_pose(new PointCloud());
      PointCloudPtr reconstruction_window(new PointCloud());
      TransformCloud(reconstruction_window_unaligned, alignment_transform, reconstruction_window);

      // Determine completeness.
      ComputePointCompleteness(scan_window, scan_window_point_indices,
		          reconstruction_window, reconstruction_window_point_indices,
                          tolerances, &first_completeness_tolerance_index, 
			  &scan_nn_point_distances, &scan_nn_point_indices,
                          output_point_completeness ? &point_is_complete : nullptr);

      // Determine accuracy.
      ComputePointAccuracy(reconstruction_window, reconstruction_window_point_indices,
		          scan_window, occupied_or_free_voxel_centers, occupied_or_free_voxel_size,
                          tolerances, &first_accuracy_tolerance_indices, 
        	          &inaccurate_classifications_exist,
                          output_point_accuracy ? &point_is_accurate: nullptr);

  }

  // Compute the mIoU over all complete points
  if (evaluate_semantic){
      std::cout << "Computing full semantic" << std::endl;
      //for (int k=0; k<scan_nn_point_indices[0].size(); k+=5000){
      //    cout << k << " " << scan_nn_point_indices[0][k] << endl;
      //}
      ComputeSemanticMeanIoU(scan_semantic, scan_confidence, reconstruction_semantic,  
		      scan_nn_point_indices, semantic_class_count, &semantic_results);

      std::ostringstream result_output_semantic_path;
      result_output_semantic_path << result_output_path << "/semantic.txt";
      WriteResult(result_output_semantic_path.str(), tolerances, semantic_results);

      std::cout << "Semantic: ";
      for (size_t tolerance_index = 0; tolerance_index < tolerances.size();
           ++tolerance_index) {
        std::cout << semantic_results[tolerance_index];
        if (tolerance_index < tolerances.size() - 1) {
          std::cout << " ";
        }
      }
      std::cout << std::endl;
  }


  // Compute the average completeness over all voxels
  std::cout << "Computing full completeness" << std::endl;
  ComputeVoxelCompleteness(scan, voxel_size_inv,
                      tolerances, first_completeness_tolerance_index,
                      &completeness_results);
  std::cout << "Completenesses at voxel level: ";
  for (size_t tolerance_index = 0; tolerance_index < tolerances.size();
       ++tolerance_index) {
    std::cout << completeness_results[tolerance_index];
    if (tolerance_index < tolerances.size() - 1) {
      std::cout << " ";
    }
  }
  std::cout << std::endl;
  ComputeAverageCompleteness(scan, 
                      tolerances, first_completeness_tolerance_index,
                      &completeness_results);
  std::cout << "Completenesses at point level: ";
  for (size_t tolerance_index = 0; tolerance_index < tolerances.size();
       ++tolerance_index) {
    std::cout << completeness_results[tolerance_index];
    if (tolerance_index < tolerances.size() - 1) {
      std::cout << " ";
    }
  }
  std::cout << std::endl;
  std::ostringstream result_output_completeness_path;
  result_output_completeness_path << result_output_path << "/completeness.txt";
  WriteResult(result_output_completeness_path.str(), tolerances, completeness_results);


  std::cout << "Computing full accuracy" << std::endl;
  ComputeVoxelAccuracy(*reconstruction, 
                  first_accuracy_tolerance_indices,
		  inaccurate_classifications_exist,
		  voxel_size_inv,
                  tolerances, 
                  &accuracy_results);
  std::cout << "Accuracies at voxel level: ";
  for (size_t tolerance_index = 0; tolerance_index < tolerances.size();
       ++tolerance_index) {
    std::cout << accuracy_results[tolerance_index];
    if (tolerance_index < tolerances.size() - 1) {
      std::cout << " ";
    }
  }
  std::cout << std::endl;
  ComputeAverageAccuracy(*reconstruction, 
                  first_accuracy_tolerance_indices,
		  inaccurate_classifications_exist,
                  tolerances, 
                  &accuracy_results);
  std::cout << "Accuracies at point level: ";
  for (size_t tolerance_index = 0; tolerance_index < tolerances.size();
       ++tolerance_index) {
    std::cout << accuracy_results[tolerance_index];
    if (tolerance_index < tolerances.size() - 1) {
      std::cout << " ";
    }
  }
  std::cout << std::endl;
  std::ostringstream result_output_accuracy_path;
  result_output_accuracy_path << result_output_path << "/accuracy.txt";
  WriteResult(result_output_accuracy_path.str(), tolerances, accuracy_results);

  // Write completeness visualization, if requested.
  if (output_point_completeness) {
    std::cout << "Writing completeness" << std::endl;
    boost::filesystem::create_directories(
        boost::filesystem::path(completeness_cloud_output_path).parent_path());
    WriteCompletenessVisualization(completeness_cloud_output_path, ground_truth_poses,
                                   scan, tolerances, point_is_complete);
  }

  // Write accuracy visualization, if requested.
  if (output_point_accuracy) {
    std::cout << "Writing accuracy" << std::endl;
    boost::filesystem::create_directories(
        boost::filesystem::path(accuracy_cloud_output_path).parent_path());
    WriteAccuracyVisualization(accuracy_cloud_output_path, *reconstruction,
                               tolerances, point_is_accurate);
  }


  std::cout << "Completenesses: ";
  for (size_t tolerance_index = 0; tolerance_index < tolerances.size();
       ++tolerance_index) {
    std::cout << completeness_results[tolerance_index];
    if (tolerance_index < tolerances.size() - 1) {
      std::cout << " ";
    }
  }
  std::cout << std::endl;
  std::cout << "Accuracies: ";
  for (size_t tolerance_index = 0; tolerance_index < tolerances.size();
       ++tolerance_index) {
    std::cout << accuracy_results[tolerance_index];
    if (tolerance_index < tolerances.size() - 1) {
      std::cout << " ";
    }
  }
  std::cout << std::endl;

  // Balanced F-score putting the same weight on accuracy and completeness, see:
  // https://en.wikipedia.org/wiki/F1_score
  std::cout << "F1-scores: ";
  for (size_t tolerance_index = 0; tolerance_index < tolerances.size();
       ++tolerance_index) {
    float precision = accuracy_results[tolerance_index];
    float recall = completeness_results[tolerance_index];
    float f1_score = (precision <= 0 && recall <= 0)
                         ? 0
                         : (2 * (precision * recall) / (precision + recall));
    std::cout << f1_score;
    if (tolerance_index < tolerances.size() - 1) {
      std::cout << " ";
    }
  }
  std::cout << std::endl;
  return static_cast<int>(ReturnCodes::kSuccess);
}
