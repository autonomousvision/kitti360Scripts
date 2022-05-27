#include "utils.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>

/**
 * @brief RunExtractCloud: Extract point cloud given indices
 * @param cloud: Input point cloud
 * @param indices: Point indices
 * @param cloud_out: Output point cloud
 * @param setnegative: if true, extract the point not in the indices
 */
void RunExtractCloud(const PointCloudPtr &cloud, vector<int> indices,
                     PointCloudPtr &cloud_out, bool setnegative){
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
    inliers->indices = indices;

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (cloud);
    extract.setIndices (inliers);
    extract.setNegative (setnegative);
    extract.filter (*cloud_out);
}

/**
 * @brief LoadConfidenceBinary Load semantic label from binary file.
 * @param filename
 * @param confidence Output confidence
 * @return length of loaded timestamp.
 */
int LoadConfidenceBinary(const std::string& filename, vector<float> &confidence) {
    confidence.clear();

    ifstream infile;
    infile.open(filename.c_str(), ios::in | ios::binary);

    if (!infile) {
	std::cout << "Failed to open timestamp data " << filename << std::endl;
        return -1;
    }

    float t;

    while (!infile.eof()) {
        infile.read((char*) &t, sizeof(t));
        if (infile.eof()) break;

        confidence.push_back(t);
    }

    infile.close();
    return confidence.size();
}

/**
 * @brief LoadSemanticLabel Load semantic label from file.
 * @param filename
 * @param label Output semantic label
 * @return length of loaded timestamp.
 */
int LoadSemanticLabel(const std::string& filename, vector<int> &label) {
    label.clear();

    ifstream infile;
    infile.open(filename.c_str());

    if (!infile) {
	std::cout << "Failed to open timestamp data " << filename << std::endl;
        return -1;
    }

    int t;

    while (!infile.eof()) {
        infile >> t;
        if (infile.eof()) break;

        label.push_back(t);
    }

    infile.close();
    return label.size();
}
