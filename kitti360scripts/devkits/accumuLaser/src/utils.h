#ifndef UTILS_H_
#define UTILS_H_

void SparsifyData (double *M,int32_t M_num,int32_t dim,int32_t* idx_sparse,int32_t &idx_size,double min_dist,double out_dist,int32_t idx_start);
void ExtractCols(Eigen::MatrixXd &input, Eigen::MatrixXd &output, int32_t *idx_sparse, int32_t idx_size);
void ExtractCols(Eigen::MatrixXd &input, Eigen::MatrixXd &output, vector<int> idx);
void ExtractRows(Eigen::MatrixXd &input, Eigen::MatrixXd &output, vector<int> idx);

void RemoveBlindSpot(Eigen::MatrixXd &matIn, Eigen::MatrixXd &matOut, float blind_splot_angle);
void CropVelodyneData (Eigen::MatrixXd &matIn, Eigen::MatrixXd &matOut, float minDist, float maxDist);
void CropVelodyneData (Eigen::MatrixXd &matIn, vector<int> &idxOut, float minDist, float maxDist);
void CurlVelodyneData (Eigen::MatrixXd &velo_in, Eigen::MatrixXd &velo_out, Eigen::Vector3d r, Eigen::Vector3d t);

void GetHalfPoints(Eigen::MatrixXd &matIn, Eigen::MatrixXd &matOut, bool direction);
void GetColorToHeight(Eigen::MatrixXd &pose, Eigen::MatrixXd &color);


#endif //UTILS_H_