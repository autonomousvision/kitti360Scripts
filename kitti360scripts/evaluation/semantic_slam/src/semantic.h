#pragma once

#include <map>
#include <vector>
#include <iostream>

void ComputeSemanticMeanIoU(const std::vector<int> scan_semantic, 
		            const std::vector<float> scan_confidence, 
		            const std::vector<int> reconstruction_semantic, 
		            const std::vector<std::vector<int>> &scan_nn_point_indices,
			    int semantic_class_count,
                            // Indexed by: [tolerance_index]. Range: [0, 1].
                            std::vector<float>* results);

