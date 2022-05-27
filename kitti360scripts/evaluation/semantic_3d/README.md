###########################################################################

                  KITTI-360: THE 3D SCENE UNDERSTANDING BENCHMARK          
                     Yiyi Liao     Jun Xie     Andreas Geiger             
                            University of Tübingen                        
           Max Planck Institute for Intelligent Systems, Tübingen         
                      www.cvlibs.net/datasets/kitti-360                   

###########################################################################


# Segmentation and Bounding Box Detection 

This section describes the KITTI-360 3D scene understanding benchmark that consists of 42 test windows for evaluating 3D semantic segmentation, 3D instance segmentation as well as 3D bounding box detection. These tasks are evaluated on the `test1` split of KITTI-360.

## Train/Val Data ##

The accumulated point clouds for training and validation as well as the training/validation split can be found in 
```
Download -> 3D data & labels -> Accumulated Point Cloud for Train & Val (128G)
```
Note that the accumulated point clouds are released as a set of batches which we refer to as `windows`. A single window contains observations within a driving distance of about 200 meters (240 frames on average) and there is an overlap of 10 meters between two consecutive windows. We provide the following information for each 3D point of the accumulated point cloud:
```
- float x 
- float y
- float z
- uchar red
- uchar green
- uchar blue
- int semantic
- int instance
- uchar visible
- float confidence
```
where `x y z` is the location of a 3D point in the world coordinate, `red green blue` is the color of a 3D point obtained by projecting it to adjacent 2D images, `semantic instance` describes the label of a 3D point. `visible` is 0 if not visible in any of the 2D frames and 1 otherwise. `confidence` is the confidence of the 3D label. Please refer to the "3D Data Format" of our online [documentation](http://www.cvlibs.net/datasets/kitti-360/documentation.php) for more details of the data format. 

The training and validation split can be found in TODO


## Test Data ##

We evaluate on 42 windows from two test sequences, 0008 and 0018. The accumulated point cloud of the test windows can be found in:
```
Download -> 3D data & labels -> Test Semantic (1.2G)
```
We provide the following information for the test windows:
```
- float x 
- float y
- float z
- uchar red
- uchar green
- uchar blue
- uchar visible
```
Compared to the training & validation windows, `semantic`, `instance` and `confidence` are held out on the evaluation server.


## Output for 3D Semantic Segmentation ##

The output structure should be analogous to the input.
All results must be provided in the root directory of a zip file using the format of npy. The file names should follow `{seq:0>4}_{start_frame:0>10}_{end_frame:0>10}.npy`. Here is how the semantic predictions should look like in root directory of your zip file. 
```
0008_0000000002_0000000245.npy
0008_0000000235_0000000608.npy
...
0018_0000000002_0000000341.npy
0018_0000000330_0000000543.npy
...
```
Each numpy file should contain a vector of the semantic labels corresponding to the accumulated point cloud. Given `N` points in the accumulated point cloud, the submitted .npy file should contain __only__ a vector of length `N`, where the `i`th scalar indicates the semantic label of the `i`th point in the accumulated point cloud. Please save your predictions in the format of __uint8__, otherwise the zip file will be too large to be accepted by the evaluation server.

The semantic labels should follow the definition of [labels.py](https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py). Note that `id` should be used instead of `kittiId` or `trainId`.

## Output for 3D Instance Segmentation ##

The output structure should be analogous to the input.
All results must be provided in the root directory of a zip file using the format of npy. The file names should follow `{seq:0>4}_{start_frame:0>10}_{end_frame:0>10}.txt` or `{seq:0>4}_{start_frame:0>10}_{end_frame:0>10}.npy`. Here is how the instance predictions should look like in root directory of your zip file. 
```
0008_0000000002_0000000245.txt
0008_0000000002_0000000245.npy
0008_0000000235_0000000608.txt
0008_0000000235_0000000608.npy
...
0018_0000000002_0000000341.txt
0018_0000000002_0000000341.npy
0018_0000000330_0000000543.txt
0018_0000000330_0000000543.npy
...
```
Each txt file should specify the specify the class of one instance and its corresponding confidence in each line as follows:
```
labelIDPrediction1 confidencePrediction1
labelIDPrediction2 confidencePrediction2
labelIDPrediction3 confidencePrediction3
...
```
For example, the 0008_0000000002_0000000245.txt may contain:
```
26 0.976347
26 0.973782
...
```
To avoid submitting a very large file to the evaluation server, we only accept predictions that assign a single instance label to each 3D point. This means the instance masks should be saved in a single vector of length `N`, where `N` is the total amount of 3D points. Each element in this vector denotes an instance ID. Let `M` denote the number of lines in the txt file, then a valid instance ID should be in the range of `1` to `M`. An instance ID of `0` means that the 3D point does not belong to any instance listed in the txt file. 

The semantic labels should follow the definition of [labels.py](https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py). Note that `id` should be used instead of `kittiId` or `trainId`. Further note that we only evaluate two classes, __building__ and __car__ for 3D instance segmentation.


## Output for 3D Bounding Box Detection ##

The output structure should be analogous to the input.
All results must be provided in the root directory of a zip file using the format of npy. The file names should follow `{seq:0>4}_{start_frame:0>10}_{end_frame:0>10}.npy`. Here is how the 3D bounding box predictions should look like in root directory of your zip file. 
```
0008_0000000002_0000000245.npy
0008_0000000235_0000000608.npy
...
0018_0000000002_0000000341.npy
0018_0000000330_0000000543.npy
...
```
Each npy file should contain a set of bounding boxes with each line denoting a bounding box parameterized as follows:
```
center_x, center_y, center_z, size_x, size_y, size_z, heading_angle, id
```
Please check the `param2Bbox` function for more details.

The semantic labels should follow the definition of [labels.py](https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py). Note that `id` should be used instead of `kittiId` or `trainId`. Further note that we only evaluate two classes, __building__ and __car__ for 3D bounding box detection.

# 3D Semantic Scene Completion 

We evaluate 3D semantic scene completion on the `test2` split of KITTI-360 contatining 38 test windows. To avoid submitting a large file to the evaluation server, we evaluate on the middle frame of each test window.

## Train/Val Data ##

In contrast to the scene perception tasks considered above, the semantic scene completion task additionally requires predicting geometry and semantics in unobserved regions. This task takes as a single LiDAR scan as input and requires semantic scene completion within a corridor of 30m around the vehicle poses of a 100m trajectory. You are free to use all data of the KITTI-360 training split for training. Please refer to the `generateCroppedGroundTruth` function for preparing the training GT given an input LiDAR scan.

## Test Data ##

The 38 frames we use for evaluation can be found at: 
```
Download -> 3D data & labels -> Test Completion (35M)
```
Each file contains a raw input point cloud defined in the world coordinate where we accumulate the point clouds.


## Output for 3D Scene Completion ##

All results must be provided in the root directory of a zip file using the format of npy. The file names should follow `{seq:0>4}_{frame:0>10}.npy`. Here is how the 3D scene completion results should look like in root directory of your zip file. 
```
0002_0000016048.npy
...
0004_0000000144.npy
...
0008_0000001602.npy
...
```
Each npy file should contain a point cloud defined in the world coordinate where the accumulated point cloud is defined. Each line of the npy file corresponds to a point location `x y z` in the world coordinate and its corresponding semantic label.
The semantic labels should follow the definition of [labels.py](https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py). Note that `id` should be used instead of `kittiId` or `trainId`.
