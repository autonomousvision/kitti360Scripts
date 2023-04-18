#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#include <pcl/registration/icp.h>
#include "util.h"
#include "commons.h"

#pragma once

using namespace Eigen;

enum class Mode {
  SE3 = 0,
  Sim3 = 1
};

void AlignTrajectory(std::vector<Matrix4f> ground_truth_poses,
		     std::vector<Matrix4f> estimated_poses, 
		     std::vector<Matrix4f> &aligned_estimated_poses,
		     Matrix4f &transform,
		     Mode mode=Mode::SE3);
