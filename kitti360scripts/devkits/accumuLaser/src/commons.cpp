#include "commons.h"
#include <fstream>
#include <math.h>
#include <stdint.h>
#include <sys/stat.h>

#define KOGMO_TIMESTAMP_TICKSPERSECOND 1000000000.0


/**
 * @brief LoadTransform Load a single transform from the file
 * @param filename
 * @param transform 4*4 matrix from file
 * @return True if loading is successful.
 */
bool LoadTransform(const char* filename, Eigen::Matrix4d &transform) {
    ifstream infile;
    infile.open(filename);

    if (!infile) {
       cout << "Failed to open transforms" << filename << endl;
       return false;
    }

    transform = Eigen::Matrix4d::Identity();

    for (int i = 0; i < 12; i ++) {
        int xi = i/4;
        int yi = i%4;
        infile >> transform(xi, yi);
    }

    infile.close();
    return true;
}


/**
 * @brief LoadCamPose Load poses from file.
 * @param filename
 * @param poses A vector of 4x4 matrix as the poses.
 * @return True if loading is successful.
 */
bool LoadCamPose(const char* filename, vector<Eigen::Matrix4d> &poses) {
    ifstream infile;
    infile.open(filename);
    if (!infile) {
       cout << "Failed to open poses" << filename << endl;
       return false;
    }

    string trash;
    poses.resize(4);

    while (!infile.eof()) {
        infile >> trash;
        if (infile.eof()) break;
        if (trash.find("image_0") != string::npos) {
            Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
            int index = atoi(trash.substr(7, 1).c_str());
            for (int i = 0; i < 12; i ++) {
                int yi = i/4;
                int xi = i%4;
                if (!infile.eof()) {
                    infile >> transform(yi, xi);
                }
            }
            poses[index] = transform;
        }
    }
    infile.close();
    return true;
}


/**
 * @brief LoadCamPose Load poses from file.
 * @param filename
 * @param poses A vector of 4x4 matrix as the poses.
 * @param index Valid indices of the poses.
 * @return True if loading is successful.
 */
bool LoadPoses(const char* filename, vector<Eigen::Matrix4d> &poses,
               vector<int> &index) {
    ifstream infile;
    infile.open(filename);

    if (!infile) {
       cout << "Failed to open poses" << filename << endl;
       return false;
    }

    poses.clear();
    index.clear();

    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    while (!infile.eof()) {
        int indx;
        infile >> indx;
        for (int i = 0; i < 12; i ++) {
            int yi = i/4;
            int xi = i%4;
            infile >> transform(yi, xi);
        }

        if (infile.eof()) break;
        poses.push_back(transform);
        index.push_back(indx);
    }

    infile.close();
    assert(poses.size() == index.size());
    return true;
}

/**
 * @brief LoadTimestamp Load timestamp from file.
 * @param filename
 * @param timestamps A vector of double values as the timestamps.
 * @return True if loading is successful.
 */
bool LoadTimestamp(const char* filename, vector<double> &timestamps){
    ifstream infile(filename);

    if (!infile) {
       cout << "Failed to open timestamps" << filename << endl;
       return false;
    }

    timestamps.clear();
    string line;

    while (getline(infile, line)){
        double ts = String2Timestamp(line.c_str());
        if (ts==0){
            cout << "Invalid timestamp at line " << line << endl;
            return false;
        }
        timestamps.push_back(ts);
    }

    return true;

}

/**
 * @brief String2Timestamp Convert timestamp in string to a double value.
 * @param time_str Timestamp in string format.
 * @return Timestamp in double value.
 */
double String2Timestamp (const char *time_str) {

  struct tm tmx;
  char ns_string[10] = "";
  memset (&tmx,0,sizeof(struct tm));

  // scan string
  int32_t err = sscanf (time_str, "%d-%d-%d%*[ _tT]%d:%d:%d.%9s",
                        &tmx.tm_year, &tmx.tm_mon, &tmx.tm_mday,
                        &tmx.tm_hour, &tmx.tm_min, &tmx.tm_sec,
                        ns_string);

  // we need at least a date (time will be 00:00:00.0 then)
  if (err<3) {
    return 0;
  }

  // the ranges of those value are a horrible confusion, see mktime(3)
  tmx.tm_year  -= +1900;
  tmx.tm_mon   -= +1;
  tmx.tm_isdst  = -1;
  time_t secs   = mktime(&tmx);
  if (secs<0) {
    return 0;
  }
  char *ns_string_end;
  uint64_t ns = strtoul (ns_string,&ns_string_end,10);

  // calculate the correct decimal fraction (9 digits)
  // this is: ns *= pow ( 10, 9 - strlen (ns_string) );
  // but prevent dependency from math-library
  for (int32_t i=ns_string_end-ns_string; i<9; i++)
    ns *= 10;

  double time_num = (double)ns + (double)secs * KOGMO_TIMESTAMP_TICKSPERSECOND;
  return time_num;

}


/**
 * @brief ReadMatrixCol Read a matrix from file with specified number of columns.
 * @param filename Filename of the matrix
 * @param matrix Eigen matrix with loaded value
 * @return True if loading is successful.
 */
bool ReadMatrixCol(const char *filename, const int cols, Eigen::MatrixXd &matrix)
{

    // Read numbers from file into buffer.
    ifstream infile;
    infile.open(filename, ios::binary);

    if (!infile) {
       cout << "Failed to open file" << filename << endl;
       return false;
    }

    infile.seekg(0, infile.end);
    size_t length = infile.tellg()/sizeof(float);
    infile.seekg(0, infile.beg);

    float* buffer = new float[length];
    infile.read( (char*)buffer, length*sizeof(float) );
    infile.close();

    if ( length % cols != 0 ){
        cout << "Unmatched number of columns in file " << filename << endl;
        return false;
    }
    int rows = length / cols;

    infile.close();

    // Populate matrix with numbers.
    matrix.resize(rows,cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix(i,j) = buffer[ cols*i+j ];

    return true;
};

/**
 * @brief WriteMatrixToFile Write Eigen matrix to file
 * @param name Filename
 * @param mat Eigen matrix
 */
void WriteMatrixToFile(const char *name, Eigen::MatrixXd &mat) {
    ofstream outfile;
    outfile.open(name);
    outfile.precision(4);

    int rows = mat.rows();
    int cols = mat.cols();
    for (int i = 0; i < rows; i ++) {
        for (int j = 0; j < cols; j++) {
            outfile << std::fixed << mat(i,j) << " ";
        }
        outfile << "\n";
    }
    outfile.close();
}

/**
 * @brief WritePoseToFile Write pose to file
 * @param name Filename
 * @param idx Vector of frame numbers for all poses
 * @param poses Vector of poses
 */
void WritePoseToFile(const char *name, vector<int> idx, vector<Eigen::Matrix4d> &poses) {
    ofstream outfile;
    outfile.open(name);
    outfile.precision(4);

    int num = idx.size();

    for (int i = 0; i < num; i ++) {
        outfile << idx[i] << " ";
        for (int l = 0; l < 16; l++) {
            int c = l/4;
            int r = l%4;
            outfile << std::fixed << poses[i](c,r) << " ";
        }
        outfile << "\n";
    }
    outfile.close();
}

/**
 * @brief WriteTimestampToFile Write timestamp to file
 * @param name Filename
 * @param timestamp Vector of timestamps
 */
void WriteTimestampToFile(const char *name, vector<int> timestamp){
    ofstream outfile;
    outfile.open(name);

    int num = timestamp.size();
    for (int i=0; i<num; i++){
        outfile << timestamp[i] << "\n";
    }
}

/**
 * @brief _mkdir Create directories recursively
 * @param dir Directory name to be created
 * @return True if succeed or if dir already exists
 */
bool _mkdir(const char *dir) {

    char tmp[256];
    char *p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp),"%s",dir);
    len = strlen(tmp);
    if(tmp[len - 1] == '/')
        tmp[len - 1] = 0;
    for(p = tmp + 1; *p; p++){
        if(*p == '/') {
                *p = 0;
                mkdir(tmp, S_IRWXU);
                *p = '/';
        }
    }
    mkdir(tmp, S_IRWXU);

    // check if dir is created afterwards so that
    // it returns true if the dir alreay exist 
    struct stat info;
    if( stat( dir, &info ) != 0 )
        return false;       // cannot access
    else if( info.st_mode & S_IFDIR )
        return true;        // dir exists
    else
        return false;       // dir does not exist

}