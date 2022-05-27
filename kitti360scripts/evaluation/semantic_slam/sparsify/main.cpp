#include <string.h>

#include "utils.h"

using namespace std;

void sparsify (PointCloudPtr &cloud, int32_t* idx_sparse, int32_t idx_size, double min_dist) {
  
  // create kdtree
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud (cloud);

  // for all data points do
  for (int32_t i=0; i<cloud->points.size(); i++) {
    
      // Neighbors within radius search
      vector<int> result;
      vector<float> dist;

      kdtree.radiusSearch(cloud->points[i], min_dist, result, dist);

      bool neighbor_exists = false;
      for (int32_t j=1; j<result.size(); j++) // exclude the first point (itself)
      {
        neighbor_exists |= (bool)idx_sparse[result[j]];
      }

      if (!neighbor_exists) {
        idx_sparse[i] = 1;
        idx_size++;
      }
  }
}


int main(int argc, char** argv) {

  // input
  std::string input_path;
  pcl::console::parse_argument(argc, argv, "--input_path",
                               input_path);
  std::string input_semantic_path;
  pcl::console::parse_argument(argc, argv, "--input_semantic_path",
                               input_semantic_path);
  std::string input_confidence_path;
  pcl::console::parse_argument(argc, argv, "--input_confidence_path",
                               input_semantic_path);
  std::string output_path;
  pcl::console::parse_argument(argc, argv, "--output_path",
                               output_path);
  std::string output_semantic_path;
  pcl::console::parse_argument(argc, argv, "--output_semantic_path",
                               output_semantic_path);
  std::string output_mask_path;
  pcl::console::parse_argument(argc, argv, "--output_mask_path",
                               output_mask_path);
  double min_dist = 0.1;
  pcl::console::parse_argument(argc, argv, "--min_dist", min_dist);

  // load input
  PointCloudPtr cloud(new PointCloud());
  if (pcl::io::loadPLYFile(input_path, *cloud) < 0) {
    std::cerr << "Cannot read reconstruction file." << std::endl;
    return static_cast<int>(0);
  }
  int32_t M_num = cloud->points.size();
  std::cout << "Loaded point cloud of " << M_num << " points" << endl;

  // sparsify
  PointCloudPtr cloud_out(new PointCloud());
  int32_t *idx_sparse = (int32_t*)calloc(M_num,sizeof(int32_t));
  int32_t  idx_size   = 0;

  sparsify(cloud,idx_sparse, idx_size, min_dist);

  vector<int32_t> idx_vector;
  idx_vector.clear();
  for (int32_t i=0; i<M_num; i++){
     if (idx_sparse[i]) idx_vector.push_back(i);
  }
  std::cout << "Keep " << idx_vector.size() << " points after filtering" << endl;

  // extract filtered point cloud
  RunExtractCloud(cloud, idx_vector, cloud_out, false);

  vector<int> semantic_label, semantic_label_filtered;
  if (!input_semantic_path.empty()) {
      LoadSemanticLabel(input_semantic_path, semantic_label);
      std::cout << "Loaded semantic label of " << semantic_label.size() << " points" << endl;
      RunExtractVector(semantic_label, idx_vector, semantic_label_filtered);
      std::cout << "Keep " << semantic_label_filtered.size() << " semantic points after filtering" << endl;
  }

  // output
  std::ostringstream file_path;
  file_path << output_path;
  pcl::io::savePLYFileBinary(file_path.str(), *cloud_out);

  if (!output_semantic_path.empty() && !input_semantic_path.empty()) {
     WriteData(output_semantic_path, semantic_label_filtered);
  }

  if (!output_mask_path.empty()) {
     WriteData(output_mask_path, idx_vector);
  }

  free(idx_sparse);

  return static_cast<int>(1);
}
