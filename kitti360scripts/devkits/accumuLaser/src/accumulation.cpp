#include "accumulation.h"
#include "commons.h"
#include "utils.h"
#include <opencv/cv.h>
#include <opencv2/core/eigen.hpp>

#define QD_BUFFER 5000

bool PointAccumulation::CreateOutputDir(){    
    printf("Output directory: %s\n", outputPath);
    return _mkdir(outputPath);
}

bool PointAccumulation::LoadTransformations(){


    // load cam0x -> pose
    string cam2poseName = calibDir + "/calib_cam_to_pose.txt";
    if (!LoadCamPose(cam2poseName.c_str(), Tr_cam_pose)) {
        return false;
    }

    // load cam00 -> velo
    string extrinsicName = calibDir + "/calib_cam_to_velo.txt";
    if (!LoadTransform(extrinsicName.c_str(), Tr_cam_velo)) {
        return false;
    }

    // load sick -> velo
    string sick2veloName = calibDir + "/calib_sick_to_velo.txt";
    if (!LoadTransform(sick2veloName.c_str(), Tr_sick_velo)) {
        return false;
    }

    // load poses
    string poseName = poseDir + "/poses.txt";
    if (!LoadPoses(poseName.c_str(), Tr_pose_world, Tr_pose_index)) {
        return false;
    }

    // calculate velo -> pose
    Tr_velo_pose = Tr_cam_pose[0] * Tr_cam_velo.inverse();
    
    return true;
}

/**
 * @brief GetInterestedWindow Get the first and last frame for clipping points
 * @return true if succeed
 */
bool PointAccumulation::GetInterestedWindow(){

    // get first frame of window used for clipping (in terms of frame number)
    float travelDist = 0;
    Eigen::Vector3d lastPose;
    bool lastPoseUpdated = false;
    for (firstFrameWindow=firstFrame; firstFrameWindow<lastFrame; firstFrameWindow++){
        // minus 1 to be consistent with matlab version
        int frameIndex = GetFrameIndex(firstFrameWindow-1);
        if (frameIndex>=0) {
            if (lastPoseUpdated){
                Eigen::Vector3d currPose = Tr_pose_world[frameIndex].block(0,3,3,1);
                currPose -= lastPose;
                travelDist += currPose.norm();
                if (travelDist > travelPadding){
                    break;
                }
            }
            lastPose = Tr_pose_world[frameIndex].block(0,3,3,1);
            lastPoseUpdated = true;
        }
    }

    // get last frame of window used for clipping (in terms of frame number)
    travelDist = 0;
    lastPoseUpdated = false;
    for (lastFrameWindow=lastFrame; lastFrameWindow>-1; lastFrameWindow--){
        // minus 1 to be consistent with matlab version
        int frameIndex = GetFrameIndex(lastFrameWindow-1);
        if (frameIndex>=0) {
            if (lastPoseUpdated){
                Eigen::Vector3d currPose = Tr_pose_world[frameIndex].block(0,3,3,1);
                currPose -= lastPose;
                travelDist += currPose.norm();
                if (travelDist > travelPadding){
                    break;
                }
            }
            lastPose = Tr_pose_world[frameIndex].block(0,3,3,1);
            lastPoseUpdated = true;
        }
    }

    if (verbose){
        printf("Window of interest within frame %010d and %010d\n", firstFrameWindow, lastFrameWindow);
    }

    if (lastFrameWindow <= firstFrameWindow){
        return false;
    }

    // calculate velo poses in the world within the current window
    Tr_velo_window.clear();
    Fidx.clear();
    Vp.clear();
    for (int frame=firstFrame; frame<lastFrame+1; frame++){
        // jump over non-existing poses 
        if (frame<1) continue;
        int frameIndex = GetFrameIndex(frame);
        if (frameIndex<0) continue;
        Tr_velo_window.push_back(Tr_pose_world[frameIndex]*Tr_velo_pose);
        Fidx.push_back(frame);
        Eigen::Vector3d vehiclePose = Tr_pose_world[frameIndex].block(0,3,3,1);

        if (frame>=firstFrameWindow && frame<=lastFrameWindow){
            Vp.push_back(vehiclePose);
        }
    }    

    return true;
}

/**
 * @brief LoadTimestamps Load timestamps for sick and velodyne data
 * @return true if succeed
 */
bool PointAccumulation::LoadTimestamps(){
    string sickTsName = baseDir + "/sick_points/timestamps.txt";
    if (!LoadTimestamp(sickTsName.c_str(), sickTimestamps)){
        return false;
    }

    string veloTsName = baseDir + "/velodyne_points/timestamps.txt";
    if (!LoadTimestamp(veloTsName.c_str(), veloTimestamps)){
        return false;
    }

    return true;
}


/**
 * @brief LoadSickData Load one frame of sick data
 * @param frame Frame number to load
 * @param data Loaded sick data
 * @return true if succeed
 */
bool PointAccumulation::LoadSickData(int frame, Eigen::MatrixXd &data){
    char frameName[256];
    snprintf(frameName, 256, "%s/sick_points/data/%010d.bin", baseDir.c_str(), frame);

    int cols = 2;
    Eigen::MatrixXd tmp;
    if(!ReadMatrixCol(frameName, cols, tmp)){
        return false;
    }

    data.resize(tmp.rows(), cols+1);
    data.setZero();
    data.block(0,1,tmp.rows(),1) = -tmp.block(0,0,tmp.rows(),1);
    data.block(0,2,tmp.rows(),1) = tmp.block(0,1,tmp.rows(),1);
    return true;
}

/**
 * @brief LoadVelodyneData Load one frame of velodyne data
 * @param frame Frame number to load
 * @param blind_splot_angle Angle for blind region
 * @param data Loaded velodyne data
 * @return true if succeed
 */
bool PointAccumulation::LoadVelodyneData(int frame, float blind_splot_angle, Eigen::MatrixXd &data){
    char frameName[256];
    snprintf(frameName, 256, "%s/velodyne_points/data/%010d.bin", baseDir.c_str(), frame);

    int cols = 4;
    Eigen::MatrixXd tmp;
    if(!ReadMatrixCol(frameName, cols, tmp)){
        return false;
    }

    RemoveBlindSpot(tmp, data, blind_splot_angle);

    Eigen::MatrixXd tmp2 = data.block(0,0,data.rows(),3);
    data = tmp2;
    return true;

}




/**
 * @brief CurlParametersFromPoses Get curl parameters at a frame
 * @param frame Frame number
 * @param Tr_pose_curr Pose at the given frame
 * @param r Output rotation matrix for curl
 * @param t Output translation matrix for curl
 */
void PointAccumulation::CurlParametersFromPoses(int frame, Eigen::Matrix4d Tr_pose_curr, Eigen::Vector3d &r, Eigen::Vector3d &t){

    // use identity in case previous or next frame have no pose (vehicle stops)
    Eigen::Matrix4d Tr_pose_pose = Eigen::Matrix4d::Identity();

    if (frame==0){
        int indexNext = GetFrameIndex(frame+1);
        if (indexNext>=0){
            Tr_pose_pose = Tr_pose_world[indexNext].inverse() * Tr_pose_curr;
        }
    }
    else{
        int indexPrev = GetFrameIndex(frame-1);
        if (indexPrev>=0){
            Tr_pose_pose = Tr_pose_curr.inverse() * Tr_pose_world[indexPrev];
        }
    }

    
    Eigen::Matrix4d Tr_delta = Tr_velo_pose.inverse() * Tr_pose_pose * Tr_velo_pose;
    Eigen::Matrix3d Tr_delta_r = Tr_delta.block(0,0,3,3);

    cv::Mat Tr_delta_CV = cv::Mat::zeros(3,3, CV_64F);
    cv::Vec3d r_CV;
    cv::eigen2cv(Tr_delta_r, Tr_delta_CV);
    cv::Rodrigues(Tr_delta_CV, r_CV);
    cv::cv2eigen(r_CV, r);
    t = Tr_delta.block(0,3,3,1);

}


/**
 * @brief GetFrameIndex Search the pose index of a given frame
 * @param frame Frame number
 * @return -1 if frame has no valid pose
 *         index of the pose if there is a pose for the frame
 */
int PointAccumulation::GetFrameIndex(int frame){
    int index;
    vector<int>::iterator it;
    it = find (Tr_pose_index.begin(), Tr_pose_index.end(), frame);
    if (it == Tr_pose_index.end()){
        index = -1;
    }
    else{
        index = std::distance(Tr_pose_index.begin(), it);
    }
    return index;
}


/**
 * @brief AddQdToMd Merge point cloud Qd to the accumulated Md
 * @param Qd Point cloud at a single frame
 * @param loc Sensor locations
 * @param frame T
 */
void PointAccumulation::AddQdToMd(Eigen::MatrixXd &Qd, Eigen::MatrixXd &loc, int frame){

    Md.resize(Md_prev.rows(), Md_prev.cols()+Qd.cols());
    Md << Md_prev, Qd;
    Ll.resize(Ll_prev.rows(), Ll_prev.cols()+loc.cols());
    Ll << Ll_prev, loc;

    // sparsify Md
    int32_t *idx_sparse = (int32_t*)calloc(Md.cols(),sizeof(int32_t));
    int32_t idx_size=0;
    double *Md_pt = &Md(0);

    SparsifyData (Md_pt, Md.cols(), Md.rows(), idx_sparse, idx_size, 
                minDistDense, outDistDense, Md_prev.cols());
    
    Eigen::MatrixXd Md_sparse;
    Eigen::MatrixXd Ll_sparse;
    ExtractCols(Md, Md_sparse, idx_sparse, idx_size);
    ExtractCols(Ll, Ll_sparse, idx_sparse, idx_size);
    Md = Md_sparse;
    Ll = Ll_sparse;

    free(idx_sparse);

    int numAdded = Md.cols() - Md_prev.cols();
    for (int i=0; i<numAdded; i++){
        Ts.push_back(frame);
    }

    Md_prev = Md;
    Ll_prev = Ll;
}


/**
 * @brief AddSickPoints Accumulate sick points
 */
void PointAccumulation::AddSickPoints(){

    // init
    int k=0;

    for (int frame=firstFrame; frame<lastFrame+1; frame++){

        // jump over non-existing poses
        if (frame<1 || GetFrameIndex(frame)<0) continue;

        // init Qd
        Eigen::MatrixXd Qd(3, 0);
        Eigen::MatrixXd Qd_prev(3, 0);
        Eigen::MatrixXd loc(3, 0);
        Eigen::MatrixXd loc_prev(3, 0);
        //size_t Qd_pt = 0;
        int prev_frame;

        // add line scans from SICK to Qd
        if (k>0){
            // velodyne timestamps between two neighboring frames
            double veloTsPrev = veloTimestamps[prev_frame];
            double veloTsCurr = veloTimestamps[frame];

            // count sick frames within the range of velodyne timestamps
            int sickCount=0;
            for (int sickIdx=0; sickIdx<sickTimestamps.size(); sickIdx++){
                double sickTsCurr = sickTimestamps[sickIdx];
                if (sickTsCurr > veloTsPrev && sickTsCurr < veloTsCurr){
                    sickCount++;
                }
                if (sickTsCurr > veloTsCurr){
                    break;
                }
            }

            // add sick scans 
            if (sickCount<30){

                // split rotation and translation
                Eigen::Matrix3d rotMatrixPrev = Tr_velo_window[k-1].block(0,0,3,3);
                Eigen::Matrix3d rotMatrixCurr = Tr_velo_window[k].block(0,0,3,3);
                Eigen::Vector3d transVecPrev = Tr_velo_window[k-1].block(0,3,3,1);
                Eigen::Vector3d transVecCurr = Tr_velo_window[k].block(0,3,3,1);

                // convert to cv::Mat for rodrigues
                cv::Mat rotMatrixPrevCV = cv::Mat::zeros(3,3, CV_64F);
                cv::Mat rotMatrixCurrCV = cv::Mat::zeros(3,3, CV_64F);
                cv::eigen2cv(rotMatrixPrev, rotMatrixPrevCV);
                cv::eigen2cv(rotMatrixCurr, rotMatrixCurrCV);
                
                
                cv::Vec3d rotVecPrevCV;
                cv::Vec3d rotVecCurrCV;
                cv::Rodrigues(rotMatrixPrevCV, rotVecPrevCV);
                cv::Rodrigues(rotMatrixCurrCV, rotVecCurrCV);

                for (int sickIdx=0; sickIdx<sickTimestamps.size(); sickIdx++){
                    double sickTsCurr = sickTimestamps[sickIdx];
                    if (sickTsCurr > veloTsCurr) break;
                    if (sickTsCurr > veloTsPrev && sickTsCurr < veloTsCurr){

                        double s = (sickTsCurr - veloTsPrev)/(veloTsCurr - veloTsPrev);
                        // interpolation
                        Eigen::Vector3d transVecInterp = (1-s)*transVecPrev + s*transVecCurr;
                        cv::Vec3f rotVecInterpCV = (1-s)*rotVecPrevCV + s*rotVecCurrCV;
                        cv::Mat rotMatInterpCV = cv::Mat::zeros(3,3, CV_64F);
                        cv::Rodrigues(rotVecInterpCV, rotMatInterpCV);
                        
                        // transform from cv::Mat back to Eigen::Matrix
                        Eigen::Matrix3d rotMatInterp = Eigen::Matrix3d::Zero();
                        cv::cv2eigen(rotMatInterpCV, rotMatInterp);

                        // recombine rotation and translation after interpolation
                        Eigen::Matrix4d Tr_sick_interp = Eigen::Matrix4d::Identity();
                        Tr_sick_interp.block(0,0,3,3) = rotMatInterp;
                        Tr_sick_interp.block(0,3,3,1) = transVecInterp;
                        Tr_sick_interp = Tr_sick_interp*Tr_sick_velo;
                        rotMatInterp = Tr_sick_interp.block(0,0,3,3);
                        transVecInterp = Tr_sick_interp.block(0,3,3,1);

                        
                        // load sick data and add it to Qt
                        Eigen::MatrixXd sickData;
                        Eigen::MatrixXd locData;
                        if (LoadSickData(sickIdx, sickData)){

                            sickData = rotMatInterp*sickData.transpose();
                            sickData.colwise() += transVecInterp;
                            locData.resize(sickData.rows(), sickData.cols());
                            locData.setZero();
                            locData.colwise() += transVecInterp;

                            Qd.resize(Qd.rows(), Qd_prev.cols()+sickData.cols());
                            Qd << Qd_prev, sickData;
                            Qd_prev = Qd;

                            loc.resize(loc.rows(), loc_prev.cols()+locData.cols());
                            loc << loc_prev, locData;
                            loc_prev = loc;

                        }
                    }
                }  
            }
        }       


        // add Qd to Md
        if (Qd.cols() > 0){
            AddQdToMd(Qd, loc, frame);
        }

        if (verbose){
            printf("Processed frame (SICK) %010d with %d points\n", frame, Md.cols());
        }

        prev_frame = frame;
        k++;
    }
}

/**
 * @brief AddVelodynePoints Accumulate velodyne points
 */
void PointAccumulation::AddVelodynePoints(){

    // init
    Eigen::MatrixXd Md_prev(3,0);
    Eigen::MatrixXd Ll_prev(3,0);

    vector<Eigen::MatrixXd> Vd;
    vector<Eigen::Matrix4d> Vd_poses;
    vector<int> Vd_frames;

    Vd.clear();
    Vd_poses.clear();
    Vd_frames.clear();

    // load velodyne data (pre-loading saves a little bit of time)
    for (int frame=firstFrame; frame<lastFrame+1; frame++){
        // if (frame>20){
        //     break;
        // }

        // jump over non-existing poses
        if (frame<1) continue;
        int frameIndex = GetFrameIndex(frame);
        if (frameIndex<0) continue;

        // read point cloud (pi/8 => almost 360° view)
        // note: for the accumulation for inference we use pi (180° view)
        Eigen::MatrixXd veloData;
        LoadVelodyneData(frame, M_PI/8, veloData);

        // get crop index, curl point clouds, remove intensities
        Eigen::Vector3d currR, currT;
        CurlParametersFromPoses(frame, Tr_pose_world[frameIndex], currR, currT);

        Eigen::MatrixXd veloDataCropped;
        CropVelodyneData(veloData, veloDataCropped, 3, 80);

        vector<int> cropIdx;
        CropVelodyneData(veloDataCropped, cropIdx, 3, maxPointDist);

        Eigen::MatrixXd veloDataCurled;
        CurlVelodyneData(veloDataCropped, veloDataCurled, currR, currT);

        // save curl point clouds without crop for octomap
        // velodyne pose
        Eigen::Matrix4d Tr_curr = Tr_pose_world[frameIndex] * Tr_velo_pose;

        // Qd from Velodyne points
        Eigen::Matrix3d Tr_curr_r = Tr_curr.block(0,0,3,3);
        Eigen::Vector3d Tr_curr_t = Tr_curr.block(0,3,3,1);
        Eigen::MatrixXd veloDataFull = Tr_curr_r * veloDataCurled.transpose();
        veloDataFull.colwise() += Tr_curr_t;
	veloDataFull.transposeInPlace();
	// add "color" to points to be consistent with matlab implementation
        Eigen::MatrixXd dummyColor = Eigen::MatrixXd::Ones(veloDataFull.rows(), 3);
	Eigen::MatrixXd veloDataFullColor(veloDataFull.rows(), veloDataFull.cols()+3);
	veloDataFullColor << veloDataFull, dummyColor;
        Ap.push_back(veloDataFullColor);

        // crop point clouds
        ExtractRows(veloDataCurled, veloDataCropped, cropIdx);

        // Nx3 to 3xN
        Vd.push_back(veloDataCropped.transpose());
        Vd_poses.push_back(Tr_pose_world[frameIndex]);
        Vd_frames.push_back(frame);

        if (verbose){
            printf("Processed frame (Velodyne loading pass): %010d \n",frame);
        }
    }

    // add Velodyne points (forward driving pass;
    // forward pass not needed for inference accumulation as it looks backwards!)
    // when the min_dist_dense is small, meaning we have denser point cloud,
    // so we don't want to do the forward pass
    if (minDistDense >= 0.02){
        for (int i=0; i<Vd.size(); i++){

            // BACKWARD facing points
            Eigen::MatrixXd Pd;
            GetHalfPoints(Vd[i], Pd, 0);

            // velodyne pose
            Eigen::Matrix4d Tr_curr = Vd_poses[i] * Tr_velo_pose;

            // Qd from Velodyne points
            Eigen::Matrix3d Tr_curr_r = Tr_curr.block(0,0,3,3);
            Eigen::Vector3d Tr_curr_t = Tr_curr.block(0,3,3,1);
            Eigen::MatrixXd Qd = Tr_curr_r * Pd;
            Qd.colwise() += Tr_curr_t;
            Eigen::MatrixXd loc(Qd.rows(), Qd.cols());
            loc.setZero();
            loc.colwise() += Tr_curr_t;

            // add Qd to Md
            AddQdToMd(Qd, loc, Vd_frames[i]);

            if (verbose){
                printf("Processed frame (Velodyne forward pass) %010d with %d points\n", Vd_frames[i], Md.cols());
            }

        }
    }
    // add Velodyne points (backward driving pass)
    for (int i=Vd.size()-1; i>-1; i--){
        // FORWARD facing points
        Eigen::MatrixXd Pd;
        GetHalfPoints(Vd[i], Pd, 1);

        // velodyne pose
        Eigen::Matrix4d Tr_curr = Vd_poses[i] * Tr_velo_pose;

        // Qd from Velodyne points
        Eigen::Matrix3d Tr_curr_r = Tr_curr.block(0,0,3,3);
        Eigen::Vector3d Tr_curr_t = Tr_curr.block(0,3,3,1);
        Eigen::MatrixXd Qd = Tr_curr_r * Pd;
        Qd.colwise() += Tr_curr_t;
        Eigen::MatrixXd loc(Qd.rows(), Qd.cols());
        loc.setZero();
        loc.colwise() += Tr_curr_t;

        // add Qd to Md
        AddQdToMd(Qd, loc, Vd_frames[i]);

        if (verbose){
            printf("Processed frame (Velodyne backward pass) %010d with %d points\n", Vd_frames[i], Md.cols());
        }
    }
}


/**
 * @brief GetPointsInRange Clip points to lie within max_point_dist from any of the poses
 */
void PointAccumulation::GetPointsInRange(){

    Eigen::MatrixXd Md_out(Md.rows(), Md.cols());
    Eigen::MatrixXd Ll_out(Ll.rows(), Ll.cols());
    vector<int> Ts_out;
    Ts_out.clear();
    double maxDist2 = maxPointDist * maxPointDist;

    int k = 0;

    int numPts = Md.cols();
    int numPos = Vp.size();
    
    for (int i=0; i<numPts; i++){
        bool inRange = false;
        double mx = Md(0, i);
        double my = Md(1, i);
        double mz = Md(2, i);
        for (int p=0; p<numPos; p++){
            double px = Vp[p](0);
            double py = Vp[p](1);
            double pz = Vp[p](2);
            if ( (mx-px)*(mx-px) + (my-py)*(my-py) + (mz-pz)*(mz-pz) < maxDist2 ){
                inRange = true;
                break;
            }
        }
        if (inRange){
            Ts_out.push_back(Ts[i]);
            Md_out.block(0,k,3,1) = Md.block(0,i,3,1);
            Ll_out.block(0,k,3,1) = Ll.block(0,i,3,1);
            k++;
        }
    }

    Md = Md_out.block(0,0,3,k);
    Ll = Ll_out.block(0,0,3,k);

    Ts = Ts_out;

    cout << "Loaded " << Md.cols() << " points after range checking" << endl;

    // transpose from 3xN to Nx3
    Md.transposeInPlace();
    Ll.transposeInPlace();

}

/**
 * @brief AddColorToPoints Add color to points according to the height
 */
void PointAccumulation::AddColorToPoints(){
    
    Eigen::MatrixXd color(Md.rows(), Md.cols());
    GetColorToHeight(Md, color);
    
    Eigen::MatrixXd Md_color(Md.rows(), Md.cols()+color.cols());
    Md_color << Md, color;

    Md = Md_color;
}

/**
 * @brief WriteToFiles Write results to file
 */
void PointAccumulation::WriteToFiles(){
    // save the 3D points into file
    if (verbose){
        printf("writing points to file %s ...\n", output_file);
    }
    WriteMatrixToFile(output_file, Md);

    // write the sensor location into file
    // (only for the combination of sick and velo)
    if (sourceType==2){
        if (verbose){
            printf("writing location to file %s ...\n", output_file_loc);
        }
        WriteMatrixToFile(output_file_loc, Ll);
    }

    // save the 3D pose information
    if (verbose){
        printf("writing pose to file %s ...\n", output_file_pose);
    }
    WritePoseToFile(output_file_pose, Fidx, Tr_velo_window);

    // save the timestamp of points into file
    if (verbose){
        printf("write timestamp to file %s ...\n", output_file_timestamp);
    }
    WriteTimestampToFile(output_file_timestamp, Ts);

}
