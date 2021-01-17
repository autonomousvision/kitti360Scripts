#ifndef ACCUMULATION_H_
#define ACCUMULATION_H_

#include <string>
#include <vector>
#include <Eigen/Dense>

#include <stdio.h>
#include <iostream>

using namespace std;

class PointAccumulation {
 public:

    string rootDir;         // root directory of the laser data
    string outputDir;
    string sequenceName;    // sequence name (the laser data will be saved in root_dir/sequence)
    int firstFrame;         // starting frame number
    int lastFrame;          // ending frame number
    float travelPadding;    // padding distance (in meters)
    int sourceType;         // source of data (0: sick only, 1: velodyne only, 2: both)
    float minDistDense;     // point cloud resolution in meter, optional default = 0.02
    bool verbose;           // boolean number to indicate if display the msg, optional default = 1

    float outDistDense;     // max distance for point to any neighbor to be inlier
    float maxPointDist;     // max distance for 3d points to any of the poses

    int firstFrameWindow;
    int lastFrameWindow;

    string baseDir, calibDir, poseDir;

    // transformation matrices
    vector<Eigen::Matrix4d> Tr_cam_pose;    // cam0x -> pose
    Eigen::Matrix4d Tr_cam_velo;            // cam00 -> velo
    Eigen::Matrix4d Tr_sick_velo;           // sick -> velo
    vector<Eigen::Matrix4d> Tr_pose_world;  // global poses of the vehicle
    vector<int> Tr_pose_index;              // index of the global poses
    Eigen::Matrix4d Tr_velo_pose;           // velo -> pose

    // timestamps
    vector<double> sickTimestamps;
    vector<double> veloTimestamps;

    Eigen::MatrixXd Md;                     // Accumulated point cloud
    Eigen::MatrixXd Ll;                     // Accumulated sensor locations
    Eigen::MatrixXd Md_prev;
    Eigen::MatrixXd Ll_prev;
    vector<Eigen::Matrix4d> Tr_velo_window; // global poses of the velodyne within [firstFrame, lastFrame]
    vector<Eigen::Vector3d> Vp;             // vehicle translations
    vector<int> Ts;                         // timestamp of accumulated point cloud
    vector<int> Fidx;                       // vector of valid frames
    vector<Eigen::MatrixXd> Ap;             // vector of dense point cloud at valid frames
 

    // output file names
    char outputPath[256];
    char output_file[256];
    char output_file_timestamp[256];
    char output_file_loc[256];
    char output_file_pose[256];

    PointAccumulation(string root_dir, string output_dir, string sequence_, int first_frame, int last_frame,
                    float travel_padding, int source_, float min_dist_dense=0.02, bool verbose_=1)
    {
        // initialization
        rootDir = root_dir;
        outputDir = output_dir;
        sequenceName = sequence_;
        firstFrame = first_frame;
        lastFrame = last_frame;
        travelPadding = travel_padding;
        sourceType = source_;
        minDistDense = min_dist_dense;
        verbose = verbose_;

        outDistDense = 0.2;
        maxPointDist = 30;

        // intialize output
        Md.resize(3, 0);
        Ll.resize(3, 0);
        Md_prev.resize(3, 0);
        Ll_prev.resize(3, 0);        
        Ts.clear();
        Ap.clear();

        // directories
        baseDir = rootDir + "/data_3d_raw/" + sequenceName;
        poseDir = rootDir + "/data_poses/" + sequenceName;
        calibDir = rootDir + "/calibration";

        // output directory
        sprintf(outputPath, "%s/%s_%06d_%06d", outputDir.c_str(), sequenceName.c_str(), firstFrame, lastFrame);

        // output filenames
        if (sourceType == 2) {
            sprintf(output_file, "%s/lidar_points_all.dat", outputPath);
            sprintf(output_file_timestamp, "%s/lidar_timestamp_all.dat", outputPath);
        }
        else if (sourceType == 1){
            sprintf(output_file, "%s/lidar_points_velodyne.dat", outputPath);
            sprintf(output_file_timestamp, "%s/lidar_timestamp_velodyne.dat", outputPath);
        }
        else{
            sprintf(output_file, "%s/lidar_points_sick.dat", outputPath);
            sprintf(output_file_timestamp, "%s/lidar_timestamp_sick.dat", outputPath);
        }
        sprintf(output_file_loc, "%s/lidar_loc.dat", outputPath);
        sprintf(output_file_pose, "%s/lidar_pose.dat", outputPath);
    }    

    bool CreateOutputDir(void);

    bool LoadTransformations(void);
    bool LoadTimestamps(void);
    bool GetInterestedWindow(void);

    bool LoadSickData(int frame, Eigen::MatrixXd &data);
    bool LoadVelodyneData(int frame, float blind_splot_angle, Eigen::MatrixXd &data);
    void AddSickPoints(void);
    void AddVelodynePoints(void);

    void CurlParametersFromPoses(int frame, Eigen::Matrix4d Tr_pose_curr, Eigen::Vector3d &r, Eigen::Vector3d &t);
    int GetFrameIndex(int frame);
    void AddQdToMd(Eigen::MatrixXd &Qd, Eigen::MatrixXd &loc, int frame);
    void GetPointsInRange(void);

    void AddColorToPoints(void);

    void WriteToFiles();



};

#endif // ACCUMULATION_H_
