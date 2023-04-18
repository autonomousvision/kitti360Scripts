#include <math.h>

using namespace std;

void curl_velodyne_data (const double* velo_in,
                         double*       velo_out,
                         const int32_t pt_num,
                         const double* r,
                         const double* t) {
  
  // for all points do
  for (int32_t i=0; i<pt_num; i++) {
    
    double vx = velo_in[i*4+0];
    double vy = velo_in[i*4+1];
    double vz = velo_in[i*4+2];
    
    double s = 0.5*atan2(vy,vx)/M_PI;
    
    double rx = s*r[0];
    double ry = s*r[1];
    double rz = s*r[2];
    
    double tx = s*t[0];
    double ty = s*t[1];
    double tz = s*t[2];
    
    double theta = sqrt(rx*rx+ry*ry+rz*rz);
    
    if (theta>1e-10) {
      
      double kx = rx/theta;
      double ky = ry/theta;
      double kz = rz/theta;
      
      double ct = cos(theta);
      double st = sin(theta);
      
      double kv = kx*vx+ky*vy+kz*vz;
      
      velo_out[i*4+0] = vx*ct + (ky*vz-kz*vy)*st + kx*kv*(1-ct) + tx;
      velo_out[i*4+1] = vy*ct + (kz*vx-kx*vz)*st + ky*kv*(1-ct) + ty;
      velo_out[i*4+2] = vz*ct + (kx*vy-ky*vx)*st + kz*kv*(1-ct) + tz;
      
      
    } else {
      
      velo_out[i*4+0] = vx + tx;
      velo_out[i*4+1] = vy + ty;
      velo_out[i*4+2] = vz + tz;
      
    }
    
    // intensity
    velo_out[i*4+3] = velo_in[i*4+3];
  }
}
