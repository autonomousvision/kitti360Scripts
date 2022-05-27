
#include "pose.h" 
#include <pcl/io/ply_io.h>

using namespace Eigen;
using namespace std;

void AlignTrajectory(std::vector<Matrix4f> ground_truth_poses,
		     std::vector<Matrix4f> estimated_poses, 
		     std::vector<Matrix4f> &aligned_estimated_poses,
		     Matrix4f &transform,
		     Mode mode){
    // Align the trajectory to the ground truth in the given mode
    // Add virtual points to align rotation
    Matrix<float, 3, Dynamic> estimated_points;
    estimated_points.resize(NoChange, estimated_poses.size()*4);
    for (std::size_t i = 0; i < estimated_poses.size(); ++ i) {
      estimated_points.col(i*4+0) = estimated_poses[i].topRightCorner<3,1>();
      estimated_points.col(i*4+1) = estimated_poses[i].block<3,1>(0,0) + estimated_poses[i].topRightCorner<3,1>();
      estimated_points.col(i*4+2) = estimated_poses[i].block<3,1>(0,1) + estimated_poses[i].topRightCorner<3,1>();
      estimated_points.col(i*4+3) = estimated_poses[i].block<3,1>(0,2) + estimated_poses[i].topRightCorner<3,1>();
    }
    
    Matrix<float, 3, Dynamic> ground_truth_points;
    ground_truth_points.resize(NoChange, ground_truth_poses.size()*4);
    for (std::size_t i = 0; i < ground_truth_poses.size(); ++ i) {
      ground_truth_points.col(i*4+0) = ground_truth_poses[i].topRightCorner<3,1>();
      ground_truth_points.col(i*4+1) = ground_truth_poses[i].block<3,1>(0,0) + ground_truth_poses[i].topRightCorner<3,1>();
      ground_truth_points.col(i*4+2) = ground_truth_poses[i].block<3,1>(0,1) + ground_truth_poses[i].topRightCorner<3,1>();
      ground_truth_points.col(i*4+3) = ground_truth_poses[i].block<3,1>(0,2) + ground_truth_poses[i].topRightCorner<3,1>();
    }
    
    // Estimate transform such that: ground_truth_point =approx= transform * estimated_point
    //Matrix<float, 4, 4> transform = Eigen::umeyama(estimated_points, ground_truth_points, mode == Mode::Sim3);
    transform = Eigen::umeyama(estimated_points, ground_truth_points, mode == Mode::Sim3);

    aligned_estimated_poses.resize(estimated_points.size());
    for (std::size_t i = 0; i < estimated_poses.size(); ++ i) {
      aligned_estimated_poses[i] = transform * estimated_poses[i];
    }
}

