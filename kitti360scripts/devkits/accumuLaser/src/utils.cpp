#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h> 
#include <Eigen/Dense>
#include <algorithm>

#include "matrix.h"
#include "kdtree_nano.h"

using namespace std;


/**
 * @brief SparsifyData Sparsify a point cloud
 * @param M Input Point cloud
 * @param M_num Number of points in M
 * @param dim Dimemsion of points
 * @param idx_sparse Output binary masks indicating which point should be preserved
 * @param idx_size Output number of points to be preserved
 * @param min_dist Minimum distance for sparsifying
 * @param out_dist Distance for removing outliers
 * @param idx_start Start sparsifying from the specific point
 */
void SparsifyData (double *M,int32_t M_num,int32_t dim,int32_t* idx_sparse,int32_t &idx_size,double min_dist,double out_dist,int32_t idx_start) {
  
  // copy model data to kdtree
  PointCloud<double> cloud;
  for (int32_t m=0; m<M_num; m++) {
      cloud.pts.push_back(Point<double>((double)M[m*dim], (double)M[m*dim+1], (double)M[m*dim+2]));  }

  // build a kd tree from the model point cloud
  KDtree<double>* tree = new KDtree<double>(cloud);
 
  // for all data points do
  for (int32_t i=0; i<M_num; i++) {
    
    if (i>=idx_start) {
      
      vector<size_t> result;
      vector<double> dist;
      tree->radiusSearch(cloud.pts[i], min_dist, result, dist);

      bool neighbor_exists = false;
      for (int32_t j=0; j<result.size(); j++)
        neighbor_exists |= (bool)idx_sparse[result[j]];

      if (!neighbor_exists) {
        idx_sparse[i] = 1;
        idx_size++;
      }
      
    // simply add
    } else {
      idx_sparse[i] = 1;
      idx_size++;
    }
  }

  
  // remove outliers
  if (out_dist>0) {
    for (int32_t i=idx_start; i<M_num; i++) {

      if (idx_sparse[i] == 1) {

        vector<size_t> result;
        vector<double> dist;
        tree->radiusSearch(cloud.pts[i], out_dist, result, dist);

        int32_t num_neighbors = 0;
        for (int32_t j=1; j<result.size(); j++) {
          if (idx_sparse[result[j]]==1)
            num_neighbors++;
        }
        
        if (num_neighbors==0) {
          idx_sparse[i] = 0;
          idx_size--;
        }
      }    
    }
  }
  
  // release memory of kd tree
  delete tree;
}


/**
 * @brief ExtractCols Extract columns of a matrix given indices
 * @param input Input matrix
 * @param output Output matrix
 * @param idx_sparse Binary masks indicate which columns should be preserved
 * @param idx_size Number of columns to be preserved
 */
void ExtractCols(Eigen::MatrixXd &input, Eigen::MatrixXd &output, int32_t *idx_sparse, int32_t idx_size){

    output.resize(input.rows(), idx_size);

    int32_t k=0;
    for (int i=0; i<input.cols(); i++){
        if (idx_sparse[i]){
            for (int c=0; c<input.rows(); c++){
                output(c,k) = input(c, i);
            }
            k++;
        }
    }
}

/**
 * @brief ExtractCols Extract columns of a matrix given indices
 * @param input Input matrix
 * @param output Output matrix
 * @param idx Indices of columns to extract
 */
void ExtractCols(Eigen::MatrixXd &input, Eigen::MatrixXd &output, vector<int> idx){

    int dim=input.rows();
    output.resize(dim, idx.size());
    for (int i=0; i<idx.size(); i++){
        for (int c=0; c<dim; c++){
            output(c, i) = input(c, idx[i]);
        }
    }
}

/**
 * @brief ExtractCols Extract columns of a matrix given indices
 * @param input Input matrix
 * @param output Output matrix
 * @param idx Indices of rows to extract
 */
void ExtractRows(Eigen::MatrixXd &input, Eigen::MatrixXd &output, vector<int> idx){

    int dim=input.cols();
    output.resize(idx.size(), dim);
    for (int i=0; i<idx.size(); i++){
        for (int c=0; c<dim; c++){
            output(i, c) = input(idx[i], c);
        }
    }
}

/**
 *@brief RemoveBlindSpot Remove the points within a sector
 *@param matIn Input point cloud
 *@param matOut Output point cloud
 *@param blind_splot_angle Sector angle
 */
void RemoveBlindSpot(Eigen::MatrixXd &matIn, Eigen::MatrixXd &matOut, float blind_splot_angle){
    if (blind_splot_angle<=0){
        matOut = matIn;
        return;
    }
    Eigen::MatrixXd matInSub = matIn.block(0,0,matIn.rows(),2);
    Eigen::MatrixXd matNorm = matInSub.rowwise().norm();

    matInSub = matIn.block(0,0,matIn.rows(),1);
    Eigen::ArrayXd v = - matInSub.array() / matNorm.array();

    float angle = cos(blind_splot_angle/2);

    int k=0;
    matOut.resize(matIn.rows(), matIn.cols());
    for (int i=0; i<matIn.rows(); i++){
        if (v(i, 0)<=angle){
            matOut.block(k,0,1,matIn.cols()) = matIn.block(i,0,1,matIn.cols());
            k++;
        }
    }
    Eigen::MatrixXd matrixSlice = matOut.block(0,0,k,matIn.cols());
    matOut = matrixSlice;
}

/**
 *@brief CropVelodyneData Crop velodyne data given max and min distance
 *@param matIn Input point cloud
 *@param matOut Output point cloud
 *@param minDist Minimum distance to crop data
 *@param maxDist Maximum distance to crop data
 */
void CropVelodyneData (Eigen::MatrixXd &matIn, Eigen::MatrixXd &matOut, float minDist, float maxDist){

    Eigen::MatrixXd matNorm = matIn.rowwise().norm();

    int k = 0;
    matOut.resize(matIn.rows(), matIn.cols());
    for (int i=0; i<matIn.rows(); i++){
        if (matNorm(i,0) > minDist && matNorm(i,0) < maxDist){
            matOut.block(k,0,1,matIn.cols()) = matIn.block(i,0,1,matIn.cols());
            k++;
        }
    }
    Eigen::MatrixXd matrixSlice = matOut.block(0,0,k,matIn.cols());
    matOut = matrixSlice;

}

/**
 *@brief CropVelodyneData Crop velodyne data given max and min distance
 *@param matIn Input point cloud
 *@param matOut Output indices of preserved points
 *@param minDist Minimum distance to crop data
 *@param maxDist Maximum distance to crop data
 */
void CropVelodyneData (Eigen::MatrixXd &matIn, vector<int> &idxOut, float minDist, float maxDist){

    Eigen::MatrixXd matNorm = matIn.rowwise().norm();

    idxOut.clear();
    for (int i=0; i<matIn.rows(); i++){
        if (matNorm(i,0) > minDist && matNorm(i,0) < maxDist){
            idxOut.push_back(i);
        }
    }
}

/**
 *@brief CurlVelodyneData Transform velodyne data given curl parameters
 *@param velo_in Input velodyne data
 *@param velo_out Output velodyne data
 *@param r Rotation matrix for curl
 *@param t Translation vector for curl
 */
void CurlVelodyneData (Eigen::MatrixXd &velo_in, Eigen::MatrixXd &velo_out, Eigen::Vector3d r, Eigen::Vector3d t){
  
  // for all points do
  int dim = velo_in.cols(); // 3
  int pt_num = velo_in.rows();
  velo_out.resize(pt_num, dim);
  for (int32_t i=0; i<pt_num; i++) {
    
    double vx = velo_in(i, 0);
    double vy = velo_in(i, 1);
    double vz = velo_in(i, 2);
    
    double s = 0.5*atan2(vy,vx)/M_PI;
    
    double rx = s*r(0);
    double ry = s*r(1);
    double rz = s*r(2);
    
    double tx = s*t(0);
    double ty = s*t(1);
    double tz = s*t(2);
    
    double theta = sqrt(rx*rx+ry*ry+rz*rz);
    
    if (theta>1e-10) {
      
      double kx = rx/theta;
      double ky = ry/theta;
      double kz = rz/theta;
      
      double ct = cos(theta);
      double st = sin(theta);
      
      double kv = kx*vx+ky*vy+kz*vz;
      
      velo_out(i, 0) = vx*ct + (ky*vz-kz*vy)*st + kx*kv*(1-ct) + tx;
      velo_out(i, 1) = vy*ct + (kz*vx-kx*vz)*st + ky*kv*(1-ct) + ty;
      velo_out(i, 2) = vz*ct + (kx*vy-ky*vx)*st + kz*kv*(1-ct) + tz;
      
      
    } else {
      
      velo_out(i, 0) = vx + tx;
      velo_out(i, 1) = vy + ty;
      velo_out(i, 2) = vz + tz;
      
    }
    
    // intensity
    // velo_out[i*4+3] = velo_in[i*4+3];
  }
}

/**
 *@brief GetHalfPoints Get either forward or backward points
 *@param matIn Input matrix
 *@param matOut Output matrix
 *@param direction Bool value indicating forward (true) or backward (false) 
 */
void GetHalfPoints(Eigen::MatrixXd &matIn, Eigen::MatrixXd &matOut, bool direction){

    int k = 0;
    matOut.resize(matIn.rows(), matIn.cols());
    for (int i=0; i<matIn.cols(); i++){
        if ((direction && matIn(0,i)>0) || (!direction && matIn(0,i)<0)){
            matOut.block(0,k,matIn.rows(),1) = matIn.block(0,i,matIn.rows(),1);
            k++;
        }
    }
    Eigen::MatrixXd matrixSlice = matOut.block(0,0,matIn.rows(),k);
    matOut = matrixSlice;
}

typedef struct {
    double r,g,b;
} COLOUR;

/**
 *@brief GetColour Turn a double into a color vector
 *@param v Input value
 *@param vmin Min value for coloring
 *@param vmax Max value for coloring
 */
COLOUR GetColour(double v,double vmin,double vmax)
{
   COLOUR c = {1.0,1.0,1.0}; // white
   double dv;

   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;
   dv = vmax - vmin;

   if (v < (vmin + 0.25 * dv)) {
      c.r = 0;
      c.g = 4 * (v - vmin) / dv;
   } else if (v < (vmin + 0.5 * dv)) {
      c.r = 0;
      c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
   } else if (v < (vmin + 0.75 * dv)) {
      c.r = 4 * (v - vmin - 0.5 * dv) / dv;
      c.b = 0;
   } else {
      c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
      c.b = 0;
   }

   return(c);
}

/**
 *@brief GetColorToHeight Assign a color to each point according to its height
 *@param pose Input point cloud
 *@param color Output color 
 */
void GetColorToHeight(Eigen::MatrixXd &pose, Eigen::MatrixXd &color){
    // get height from poses
    Eigen::VectorXd height = pose.block(0,2,pose.rows(),1);

    // sort and get the quantiles
    sort(height.data(),height.data()+height.size());
    int idxL = (int)(ceil((double)height.size()*0.05));
    int idxH = (int)(floor((double)height.size()*0.95));
    double heightL = height(idxL);
    double heightH = height(idxH);

    for (int i=0; i<height.size(); i++){
        COLOUR c = GetColour(pose(i, 2), heightL, heightH);
        color(i, 0) = c.r;
        color(i, 1) = c.g;
        color(i, 2) = c.b;
    }
}