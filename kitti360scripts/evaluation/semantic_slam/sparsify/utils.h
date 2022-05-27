#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <pcl/console/parse.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;

void RunExtractCloud(const PointCloudPtr &cloud, vector<int> indices,
                     PointCloudPtr &cloud_out, bool setnegative);
int LoadConfidenceBinary(const std::string& filename, vector<float> &confidence);
int LoadSemanticLabel(const std::string& filename, vector<int> &label);

template<typename T>
void RunExtractVector(const vector<T> &input, vector<int> indices, vector<T> &output, bool inverse=false){
  output.clear();
  if (inverse){
	if (indices.size()==0){
		for (int i=0; i<input.size(); i++){
			output.push_back(input[i]);
		}
		return;
	} 
	sort(indices.begin(), indices.end()); 
	int offset = 0;
	for (int i=0;i<input.size(); i++){
		if (i==indices[offset]){
			offset++;
			continue;
		}else{
	  		output.push_back(input[i]);
		}
	}
  }else{
      for (int i=0; i<indices.size(); i++){
              output.push_back(input[indices[i]]);
      }
  }
}

template<typename T>
void WriteData(const std::string& filename, vector<T> &vector){
    ofstream outfile;
    outfile.open(filename.c_str());

    for (int i = 0; i < vector.size(); i++) {
        outfile << vector[i] << "\n";
    }
    outfile.close();
}
