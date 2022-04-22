###########################################################################

           KITTI-360: THE 2D SEMANTIC INSTANCE SEGMENATION BENCHMARK          
                    Yiyi Liao     Jun Xie     Andreas Geiger                  
                           University of Tübingen                        
          Max Planck Institute for Intelligent Systems, Tübingen         
                     www.cvlibs.net/datasets/kitti-360                   

###########################################################################



This file describes the KITTI-360 2D semantic/instance segmentation benchmark that consists of 910 test images. 


## Train/Val Data ##

The RGB images for training and validation can be found in 
```
Download -> 2D data & labels -> Perspective Images for Train & Val (128G)
```
The corresponding labels for training and validation as well as the training/validation split can be found in 
```
Download -> 2D data & labels -> Semantics (1.8G) 
```
Please refer to the "2D Data Format" of our online [documentation](http://www.cvlibs.net/datasets/kitti-360/documentation.php) for more details of the data format.


## Test Data ##

We evaluate on 910 images of the left perspective camera sampled from two test
sequences, 0008 and 0018. The RGB images for testing can be found in: ```
Download -> 2D data & labels -> Test Semantic (1.5G) ```

## Output for 2D Semantic Segmentation ##

The output structure should be analogous to the input.
All results must be provided in the root directory of a zip file using the format of 8 bit grayscale png. The file names should follow `{seq:0>4}_{frame:0>10}.png`. Here is how the semantic predictions should look like in root directory of your zip file. 
```
0008_0000000002.png
0008_0000000012.png
...
0018_0000000079.png
0018_0000000094.png
...
```
The semantic labels should follow the definition of [labels.py](https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py). Note that `id` should be used instead of `kittiId` or `trainId`.

## Output for 2D Instance Segmentation ##

For 2D instance segmentation, all results must be provided in the root directory of a zip file in the format of txt or png files. Here is how the instance predictions may look like in root directory of your zip file. 
```
0008_0000000002.txt
0008_0000000002_0000.png
0008_0000000002_0001.png
...
0018_0000000079.txt
0018_0000000079_0000.png
0018_0000000079_0001.png
```
The txt files of the instance segmentation should look as follows:
```
relPathPrediction1 labelIDPrediction1 confidencePrediction1
relPathPrediction2 labelIDPrediction2 confidencePrediction2
relPathPrediction3 labelIDPrediction3 confidencePrediction3
...
```

For example, the 0008_0000000002.txt may contain:
```
0008_0000000002_0000.png 026 0.976347
0008_0000000002_0001.png 026 0.973782
...
```
