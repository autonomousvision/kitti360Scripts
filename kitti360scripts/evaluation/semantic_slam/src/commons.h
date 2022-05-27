
#ifndef CODE_DATACONVERTER_SRC_COMMONS_H_
#define CODE_DATACONVERTER_SRC_COMMONS_H_

#include <fstream>
#include <iostream>
#include <map>

#include "util.h"

using namespace std;

int LoadDenseData(const std::string& filename, PointCloud::Ptr &outputCloud,
                  PointXYZ &center);
int LoadDenseDataXYZ(const std::string& filename, PointCloud::Ptr &outputCloud);
int LoadDenseTimestamp(const std::string& filename, vector<int> &timestamp);
int LoadDenseTimestampAscii(const std::string& filename, vector<vector<int>> &timestamp);
int LoadDenseTimestampBinary(const std::string& filename, vector<vector<int>> &timestamp);
bool LoadPoses(const std::string& filename, map<int, Eigen::Matrix4f> &scan_infos);
int LoadVoxelCenters(const std::string& filename, PointCloud::Ptr &outputCloud);
int LoadSemanticLabel(const std::string& filename, vector<int> &label);
int LoadConfidence(const std::string& filename, vector<float> &confidence);
int LoadConfidenceBinary(const std::string& filename, vector<float> &confidence);
void WriteResult(const std::string& filename, std::vector<float> tolerances, std::vector<float> result);
void WriteBinaryFile(const std::string& filename, pcl::PointCloud<pcl::PointXYZRGB> &cloud, int save_every=100 );

void GetPointsInRange(const PointCloudPtr cloud, 
		      std::vector<Eigen::Matrix4f> poses,
		      PointCloudPtr &output_cloud,
		      std::vector<int> &inlier_indices,
		      double maxPointDist=80.);
void RunExtractCloud(const PointCloudPtr cloud, pcl::PointIndices::Ptr inliers,
                     PointCloudPtr &cloud_out, bool setnegative=false);
void TransformCloud(const PointCloudPtr src_cloud, const Eigen::Matrix4f transform,
		    PointCloudPtr &tgt_cloud);
void DownsampleCloud(const PointCloudPtr src_cloud, PointCloudPtr &tgt_cloud,
		     float leaf_size=0.1);
#endif  // CODE_DATACONVERTER_SRC_COMMONS_H_
