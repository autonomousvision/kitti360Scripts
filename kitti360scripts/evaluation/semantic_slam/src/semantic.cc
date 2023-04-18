#include "semantic.h"

std::map<int, int> classmap={
    {0, 255}, // unlabeled
    {1, 255}, // ego vehicle
    {2, 255}, // rectification border
    {3, 255}, // out of roi
    {4, 255}, // static
    {5, 255}, // dynamic
    {6, 255}, // ground
    {7, 0}, // road
    {8, 1}, // sidewalk
    {9, 255}, // parking
    {10, 255}, // rail track
    {11, 2}, // building
    {12, 3}, // wall
    {13, 4}, // fence
    {14, 255}, // guard rail
    {15, 255}, // bridge
    {16, 255}, // tunnel
    {17, 5}, // pole
    {18, 255}, // polegroup
    {19, 6}, // traffic light
    {20, 7}, // traffic sign
    {21, 8}, // vegetation
    {22, 9}, // terrain
    {23, 10}, // sky
    {24, 11}, // person
    {25, 12}, // rider
    {26, 13}, // car
    {27, 14}, // truck
    {28, 15}, // bus
    {29, 255}, // caravan
    {30, 255}, // trailer
    {31, 16}, // train
    {32, 17}, // motorcycle
    {33, 18}, // bicycle
    {34, 2}, // garage
    {35, 4}, // gate
    {36, 255}, // stop
    {37, 5}, // smallpole
    {38, 255}, // lamp
    {39, 255}, // trash bin
    {40, 255}, // vending machine
    {41, 255}, // box
    {42, 255}, // unknown construction
    {43, 255}, // unknown vehicle
    {44, 255}, // unknown object
};

void ComputeSemanticMeanIoU(const std::vector<int> scan_semantic, 
		            const std::vector<float> scan_confidence, 
		            const std::vector<int> reconstruction_semantic, 
		            const std::vector<std::vector<int>> &scan_nn_point_indices,
			    int semantic_class_count,
                            std::vector<float>* results){

  int tolerances_count = scan_nn_point_indices.size();

  // initialize confusion matrix
  std::vector<std::vector<std::vector<float>>> confusion_matrices(tolerances_count);
  for (int k=0; k<tolerances_count; k++){
      confusion_matrices[k].clear();
      for (int i=0; i<semantic_class_count+1; i++){
          std::vector<float> confusion_vector(semantic_class_count+1, 0.0);
          confusion_matrices[k].push_back(confusion_vector);
      }
  }

  // loop over all points in the ground truth cloud to obtain the confusion matrix
  for (int i=0; i<scan_semantic.size(); i++){
      int gt_class_raw = scan_semantic[i];
      // map id to trainId
      int gt_class = classmap[gt_class_raw];
      if (gt_class==255){
	      gt_class = semantic_class_count;
      }
      for (int tolerance_index=0; tolerance_index<tolerances_count; tolerance_index++){
	  // there exist a point that is close enough to the ground truth point
          if (scan_nn_point_indices[tolerance_index][i]>-1){
              int pred_class_raw = reconstruction_semantic[scan_nn_point_indices[tolerance_index][i]];
              // map id to trainId
              int pred_class = classmap[pred_class_raw];
	      if (pred_class==255) pred_class=semantic_class_count;
              // confidence-weighted evaluation
              confusion_matrices[tolerance_index][gt_class][pred_class] += scan_confidence[i];
          }
	  // no reconstruction point completes the ground truth point, will be considered as false negative
	  else{
              // confidence-weighted evaluation
              confusion_matrices[tolerance_index][gt_class][semantic_class_count] += scan_confidence[i];
	  }
      }
  }

  // calculate average score
  for (int k=0; k<tolerances_count; k++){
      std::vector<float> true_positive_count(semantic_class_count, 0);
      std::vector<float> false_negative_count(semantic_class_count, 0);
      std::vector<float> false_positive_count(semantic_class_count+1, 0);
      // ignore last row, i.e., ground truth point is labeled as unknown/ignored classes
      for (int i=0; i<semantic_class_count; i++){
          for (int j=0; j<semantic_class_count+1; j++){
              //all_count +=  confusion_matrices[k][i][j];
	      if (i==j){
		  true_positive_count[i] += confusion_matrices[k][i][j];
	      }else{
	          false_negative_count[i] += confusion_matrices[k][i][j];
	          false_positive_count[j] += confusion_matrices[k][i][j];
	      }
              std::cout << confusion_matrices[k][i][j] << " ";
          }
          std::cout << std::endl;
      }

      // calculate mean IoU
      std::vector<float> score(semantic_class_count, 0);
      float mIoU = 0.0;
      float valid_count = 0.0;
      for (int i=0; i<semantic_class_count; i++){
	  float denom = true_positive_count[i] + false_positive_count[i] + false_negative_count[i];
	  if (denom>0){
	      score[i] = true_positive_count[i] / denom;
	      valid_count += 1;
	      mIoU += score[i];
	  } else{
              score[i] = -1;
	  }
	  std::cout << "class " << i << " score " << score[i] << std::endl;
      }
      mIoU /= valid_count;
      results->at(k) = mIoU;
  }
}
