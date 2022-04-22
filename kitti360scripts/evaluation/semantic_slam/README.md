###########################################################################

                     KITTI-360: THE LOCALIZATION BENCHMARK          
                    Yiyi Liao     Jun Xie     Andreas Geiger                  
                           University of Tübingen                        
          Max Planck Institute for Intelligent Systems, Tübingen         
                     www.cvlibs.net/datasets/kitti-360                   

###########################################################################



This file describes the KITTI-360 localization benchmark that consists of 4 short test sequences.


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

## Output ##

All results must be provided in the root directory of a zip file in the format of txt:
```
test_poses_0.txt
test_poses_1.txt
test_poses_2.txt
test_poses_3.txt
```
For each txt file, one line indicates a flattened 3x4 matrix. Note that our ground truth is the rigid body transform from GPU/IMU coordinates to a world coordinate system (`poses.txt`). Please check our `test_data` folder for examples.


## Evaluation ##

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
