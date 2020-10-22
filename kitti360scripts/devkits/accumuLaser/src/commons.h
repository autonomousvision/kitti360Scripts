#ifndef COMMONS_H_
#define COMMONS_H_

#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <iostream>


using namespace std;

bool LoadTransform(const char* filename, Eigen::Matrix4d &transform);
bool LoadCamPose(const char* filename, vector<Eigen::Matrix4d> &poses);
bool LoadPoses(const char* filename, vector<Eigen::Matrix4d> &poses,
               vector<int> &index);
bool LoadTimestamp(const char* filename, vector<double> &timestamps);
double String2Timestamp(const char *time_str);
bool ReadMatrixCol(const char *filename, const int cols, Eigen::MatrixXd &matrix);
void WriteMatrixToFile(const char *name, Eigen::MatrixXd &mat);
void WritePoseToFile(const char *name, vector<int> idx, vector<Eigen::Matrix4d> &poses);
void WriteTimestampToFile(const char *name, vector<int> timestamp);
bool _mkdir(const char *dir);

#endif // COMMONS_H_