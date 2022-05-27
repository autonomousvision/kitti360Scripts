###########################################################################

                     KITTI-360: THE SEMANTIC SLAM BENCHMARK          
                    Yiyi Liao     Jun Xie     Andreas Geiger                  
                           University of Tübingen                        
          Max Planck Institute for Intelligent Systems, Tübingen         
                     www.cvlibs.net/datasets/kitti-360                   

###########################################################################



This file describes the KITTI-360 semantic localization and mapping benchmark that consists of 4 short test sequences.


## Test Data ##

We evaluate all methods on 4 short sequences. For all sequences, we release both stereo images and Velodyne scans.
You are free to choose the input modality. Please indicate the input modality you used when submitting to our benchmark.
The stereo images of our test sequences are available in: 
```
Download -> 2D data & labels -> Test SLAM (14G) 
```
The Velodyne scans of our test sequences can be downlaoded in:
```
Download -> 3D data & labels -> Test SLAM (12G) 
```

As we filter out very slow-moving frames, the frame ID of the test images may not be continuous.
After filtering, there are 933, 1547, 2848, 3012 frames in our test sequences, respectively. To evaluate your method on our server, we expect exactly the same amount of frames in your submission.

# Localization

## Output for Localication ##

All results must be provided in the root directory of a zip file in the format of txt:
```
test_poses_0.txt
test_poses_1.txt
test_poses_2.txt
test_poses_3.txt
```
For each txt file, one line indicates a flattened 3x4 matrix. Note that our ground truth is the rigid body transform from GPU/IMU coordinates to a world coordinate system (`poses.txt`). Please check our `test_data` folder for examples.


## Evaluation of Localization ##

We adopt the great package [evo](https://github.com/MichaelGrupp/evo) to evaluate localization errors.
To test your performance locally, you can install evo via:
```
pip install evo --upgrade --no-binary evo
```
Next, you can test our evaluation script via:
```
./evalTrajectory.sh
```
Note that this test script compares the outputs of two methods, i.e., no ground truth is used here.

# Semantic Mapping

## Output for Semantic Mapping ##

All results must be provided in the root directory of a zip file. The file should contains the estimated poses, the reconstruction as well as the semantic labels:
```
test_poses_0.txt
test_poses_1.txt
test_poses_2.txt
test_poses_3.txt
test_0.ply
test_1.ply
test_2.ply
test_3.ply
test_semantic_0.txt
test_semantic_1.txt
test_semantic_2.txt
test_semantic_3.txt
```
For each pose txt file, one line indicates a flattened 3x4 matrix. Note that our ground truth is the rigid body transform from GPU/IMU coordinates to a world coordinate system (`poses.txt`). Please check our `test_data` folder for examples. The ply files are reconstructed point clouds containing only `x y z` locations. The semantic label of each point should be provided in a seperate txt file containing a vector of length `M` equal to the number of points in the ply file.
The semantic labels should follow the definition of [labels.py](https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py). Note that `id` should be used instead of `kittiId` or `trainId`.

*IMPORTANT* Note that our evaluation server does not accept submission file larger than 200MB. It is required to apply our downsampling script to your reconstructed point cloud before submission. Please check the [sparify](https://github.com/autonomousvision/kitti360Scripts/tree/master/kitti360scripts/evaluation/semantic_slam/sparsify) code for more details.

## Evaluation of Semantic Mapping ##

The evaluation script is adapted from the [ETH3D](https://github.com/ETH3D/multi-view-evaluation) benchmark. Here are the required dependencies:
```
Boost
Eigen3
PCL1.7
```
With the required dependencies installed, you can build the evaluation script via:
```
mkdir build && cd build
cmake ..
make
```

