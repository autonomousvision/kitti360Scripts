#include "commons.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <pcl/filters/voxel_grid.h>

/**
 * @brief LoadCamPose Load poses from file.
 * @param filename
 * @param poses A vector of 4x4 matrix as the poses.
 * @param index Valid indices of the poses.
 * @return True if loading is successful.
 */
bool LoadPoses(const std::string& filename, map<int, Eigen::Matrix4f> &scan_infos){
    scan_infos.clear();
    ifstream infile;
    infile.open(filename.c_str());

    if (!infile) {
       cout << "Failed to open poses" << filename << endl;
       return false;
    }

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    int indx = 0;
    while (!infile.eof()) {
        //int indx;
        //infile >> indx;
        for (int i = 0; i < 12; i ++) {
            int yi = i/4;
            int xi = i%4;
            infile >> transform(yi, xi);
        }

        if (infile.eof()) break;
	scan_infos.insert(pair<int, Eigen::Matrix4f>(indx, transform));
        
        indx++;
    }

    infile.close();
    return true;
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

/**
 * @brief LoadConfidence Load semantic label from file.
 * @param filename
 * @param confidence Output confidence
 * @return length of loaded timestamp.
 */
int LoadConfidence(const std::string& filename, vector<float> &confidence) {
    confidence.clear();

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

        confidence.push_back(t);
    }

    infile.close();
    return confidence.size();
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
 * @brief LoadDenseTimestamp Load dense timestamp data from file.
 * @param filename
 * @param timestamp Output timestamp of dense point
 * @return length of loaded timestamp.
 */
int LoadDenseTimestamp(const std::string& filename, vector<int> &timestamp) {
    timestamp.clear();

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

        timestamp.push_back(t);
    }

    infile.close();
    return timestamp.size();
}

/**
 * @brief LoadDenseTimestamp Load dense timestamp data from file.
 * @param filename
 * @param timestamp Output timestamp of dense point
 * @return length of loaded timestamp.
 */
int LoadDenseTimestampBinary(const std::string& filename, vector<vector<int>> &timestamp) {
    timestamp.clear();

    ifstream infile;
    infile.open(filename.c_str(), ios::in | ios::binary);

    if (!infile) {
	std::cout << "Failed to open timestamp data " << filename << std::endl;
        return -1;
    }

    int t;
    int size;

    while (!infile.eof()) {
        infile.read((char*) &size, sizeof(size));
        if (infile.eof()) break;
	vector<int> result;
	for (int i=0; i<size; i++){
            infile.read((char*) &t, sizeof(t));
	    result.push_back(t);
	}
        timestamp.push_back(result);
    }

    infile.close();
    return timestamp.size();
}

/**
 * @brief LoadDenseTimestamp Load dense timestamp data from file.
 * @param filename
 * @param timestamp Output timestamp of dense point
 * @return length of loaded timestamp.
 */
int LoadDenseTimestampAscii(const std::string& filename, vector<vector<int>> &timestamp) {
    timestamp.clear();

    ifstream infile;
    infile.open(filename.c_str());

    if (!infile) {
	std::cout << "Failed to open timestamp data " << filename << std::endl;
        return -1;
    }

    int t;

    string line;
    while (getline(infile,line)){
	    vector<int> result;
	    result.clear();
	    istringstream stm(line) ; 
	    int value ;
	    //while( stm.read((char*) &value, sizeof(value)) ) {result.push_back(value) ; 
	    while( stm >> value ) result.push_back(value) ; 
	    timestamp.push_back(result);
    }

    infile.close();
    return timestamp.size();
}

/**
 * @brief LoadDenseData Load dense xyzrgb data from file.
 * @param filename
 * @param outputCloud Output point cloud as in PCL xyzrgb format
 * @param center Center of the point cloud.
 * @return True if loading is successful.
 */
int LoadDenseData(const std::string& filename, PointCloud::Ptr &outputCloud,
                  PointXYZ &center) {
    outputCloud.reset (new PointCloud);

    ifstream infile;
    infile.open(filename.c_str());

    if (!infile) {
        std::cout << "Failed to open dense data XYZ " << filename << std::endl;
        return -1;
    }

    float r, g, b, x, y, z;
    PointXYZ pt;
    center.x = 0; center.y = 0; center.z = 0;

    while (!infile.eof()) {
        infile >> x >> y >> z >> r >> g >> b;
        if (infile.eof()) break;

        pt.x = x; pt.y = y; pt.z = z;
        outputCloud->points.push_back(pt);

        center.x += x;
        center.y += y;
        center.z += z;
    }

    infile.close();

    center.x /= outputCloud->points.size();
    center.y /= outputCloud->points.size();
    center.z /= outputCloud->points.size();

    outputCloud->width = outputCloud->points.size();
    outputCloud->height = 1;
    return outputCloud->points.size();
}

/**
 * @brief LoadDenseData Load dense xyzrgb data from file.
 * @param filename
 * @param outputCloud Output point cloud as in PCL xyzrgb format
 * @param center Center of the point cloud.
 * @return True if loading is successful.
 */
int LoadVoxelCenters(const std::string& filename, PointCloud::Ptr &outputCloud) 
{
    outputCloud.reset (new PointCloud);

    ifstream infile;
    infile.open(filename.c_str());

    if (!infile) {
        std::cout << "Failed to open dense data XYZ " << filename << std::endl;
        return -1;
    }

    float x, y, z, occupancy;
    PointXYZ pt;

    while (!infile.eof()) {
        infile >> x >> y >> z >> occupancy;
        if (infile.eof()) break;

        pt.x = x; pt.y = y; pt.z = z;
        outputCloud->points.push_back(pt);
    }

    infile.close();

    outputCloud->width = outputCloud->points.size();
    outputCloud->height = 1;
    return outputCloud->points.size();
}

/**
 * @brief GetPointsInRange Clip points to lie within max_point_dist from any of the poses
 *        such that when people annotate they don't try very far away objects
 */
void GetPointsInRange(const PointCloudPtr cloud, 
		      std::vector<Eigen::Matrix4f> poses,
		      PointCloudPtr &output_cloud,
		      std::vector<int> &inlier_indices,
		      double maxPointDist){

    //vector<int> indices;
    inlier_indices.clear();
    double maxDist2 = maxPointDist * maxPointDist;

    int k = 0;

    int numPts = cloud->points.size();
    int numPos = poses.size();

    for (int i=0; i<numPts; i++){
        bool inRange = false;
        double mx = cloud->points[i].x;
        double my = cloud->points[i].y;
        double mz = cloud->points[i].z;
        for (int p=0; p<numPos; p++){
            double px = poses[p].coeff(0,3);
            double py = poses[p].coeff(1,3);
            double pz = poses[p].coeff(2,3);
            if ( (mx-px)*(mx-px) + (my-py)*(my-py) + (mz-pz)*(mz-pz) < maxDist2 ){
                inRange = true;
                break;
            }
        }
        if (inRange){
            inlier_indices.push_back(i);
            k++;
        }
    }

    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
    inliers->indices = inlier_indices;

    RunExtractCloud(cloud, inliers, output_cloud);

    cout << "Loaded " << output_cloud->points.size() << "/" << cloud->points.size()  << " points after range checking" << endl;

}

/**
 * @brief RunExtractCloud: Extract point cloud given indices
 * @param cloud: Input point cloud
 * @param inliers: Point indices
 * @param cloud_out: Output point cloud
 * @param setnegative: if true, extract the point not in the indices
 */
void RunExtractCloud(const PointCloudPtr cloud, pcl::PointIndices::Ptr inliers,
                     PointCloudPtr &cloud_out, bool setnegative){
    pcl::ExtractIndices<PointXYZ> extract;
    extract.setInputCloud (cloud);
    extract.setIndices (inliers);
    extract.setNegative (setnegative);
    extract.filter (*cloud_out);
}

void TransformCloud(const PointCloudPtr src_cloud, const Eigen::Matrix4f transform,
		    PointCloudPtr &tgt_cloud){
    pcl::transformPointCloud (*src_cloud, *tgt_cloud, transform);
}

void DownsampleCloud(const PointCloudPtr src_cloud, PointCloudPtr &tgt_cloud, float leaf_size){
    // Create the filtering object
    pcl::VoxelGrid<PointXYZ> sor;
    sor.setInputCloud (src_cloud);
    sor.setLeafSize (leaf_size, leaf_size, leaf_size);
    sor.filter (*tgt_cloud);
}

/**
 * @brief WriteAnnotation Write depth results to file.
 * @param name
 * @param pixelDepth Pixelwise depth.
 * @param w Width of the image
 * @param h Height of the image
 */
void WriteResult(const std::string& filename, std::vector<float> tolerances, std::vector<float> result){
    ofstream outfile;
    outfile.open(filename.c_str());
    for (int i = 0; i < tolerances.size(); i ++) {
        outfile << tolerances[i] << " ";
    }
    outfile << endl;
    for (int i = 0; i < result.size(); i ++) {
        outfile << result[i] << " ";
    }
    outfile << endl;
    outfile.close();
}

void WriteBinaryFile(const std::string& filename, pcl::PointCloud<pcl::PointXYZRGB> &cloud, int save_every ){
    ofstream outfile;
    outfile.open(filename.c_str(), ios::binary);
    for (int i = 0; i < cloud.points.size(); i+=save_every) {
        float x = (float)cloud.points[i].x;
        float y = (float)cloud.points[i].y;
        float z = (float)cloud.points[i].z;
        uint8_t r = cloud.points[i].r;
        uint8_t g = cloud.points[i].g;
        uint8_t b = cloud.points[i].b;
	outfile.write( (char *) ( &x ), sizeof( x ) ); 
	outfile.write( (char *) ( &y ), sizeof( y ) ); 
	outfile.write( (char *) ( &z ), sizeof( z ) ); 
	outfile.write( (char *) ( &r ), sizeof( r ) ); 
	outfile.write( (char *) ( &g ), sizeof( g ) ); 
	outfile.write( (char *) ( &b ), sizeof( b ) ); 
    }
    outfile.close();
}
