#include <iostream>
#include <cstdlib>
#include "accumulation.h"

int main(int argc, char* argv[]){
    
    if (argc < 8){
        std::cerr << "Usage: " << argv[0] << " root_dir sequence output_dir first_frame last_frame \
            travel_padding source [min_dist_dense] [verbose]" << std::endl;
        return 1;
    }

    std::string root_dir = argv[1];
    std::string sequence = argv[2];
    std::string output_dir = argv[3];
    int first_frame = atoi(argv[4]);
    int last_frame = atoi(argv[5]);
    float travel_padding = atof(argv[6]);
    int source = atoi(argv[7]);
    float min_dist_dense = 0.02;
    bool verbose = true;

    if (argc > 8){
        min_dist_dense = atof(argv[8]);
    }
    if (argc > 9 && atoi(argv[9])<1){
        verbose = false;
    }
    

    PointAccumulation pointAcc(root_dir, output_dir, sequence, first_frame, last_frame, 
                               travel_padding, source, min_dist_dense, verbose);
    
    if (verbose) std::cout << "Initialization Done!" << std::endl;

    if (!pointAcc.CreateOutputDir()){
        std::cerr << "Error: Unable to create the output directory!" << std::endl;
        return 1; 
    }

    if (!pointAcc.LoadTransformations()){
        std::cerr << "Error: Unable to load the calibrations!" << std::endl;
        return 1;
    }

    if (!pointAcc.GetInterestedWindow()){
        std::cerr << "Error: Invalid window of interested!" << std::endl;
        return 1;
    }

    if (verbose) {
        std::cout << "Loaded " << pointAcc.Tr_pose_world.size() << " poses" << std::endl;
    }

    pointAcc.LoadTimestamps();

    if (verbose) {
        std::cout << "Loaded " << pointAcc.sickTimestamps.size() << " sick timestamps" << std::endl;
        std::cout << "Loaded " << pointAcc.veloTimestamps.size() << " velo timestamps" << std::endl;
    }

    if (source==0 || source==2){
        pointAcc.AddSickPoints();
    }

    if (source==1 || source==2){
        pointAcc.AddVelodynePoints();
    }

    pointAcc.GetPointsInRange();

    pointAcc.AddColorToPoints();

    pointAcc.WriteToFiles();


    return 0;
}