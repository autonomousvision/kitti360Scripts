###########################################################################

                 KITTI-360: THE NOVEL VIEW SYNTHESIS BENCHMARK          
                    Yiyi Liao     Jun Xie     Andreas Geiger                  
                           University of Tübingen                        
          Max Planck Institute for Intelligent Systems, Tübingen         
                     www.cvlibs.net/datasets/kitti-360                   

###########################################################################



This file describes the KITTI-360 2D semantic segmentation benchmark that consists of 910 test images. 


## Training Views ##

The RGB images for training and the train/test split can be found in:
```
Download -> 2D data & labels -> Test NVS (0.3G)
```
The camera poses can be found in:
```
Download -> Vechicle Poses (8.9M)
```
For camera poses, the first number at each row corresponds to the frame index. 
Please refer to the "2D Data Format" of our online [documentation](http://www.cvlibs.net/datasets/kitti-360/documentation.php) for more details of the data format.


## Test Data ##

We evaluate on 100 images of the left perspective camera sampled from two test
sequences, 0008 and 0018. Camera poses of the test views can be found in:
```
Download -> Vechicle Poses (8.9M)
```

## Output for Novel View Appearance Synthesis ##

The output structure should be analogous to the input.
All results must be provided in the root directory of a zip file using the format of 8 bit png. The file names should follow `{seq:0>4}_{frame:0>10}.png`. Here is how the semantic predictions should look like in root directory of your zip file. 
```
0008_0000000828.png
0008_0000000830.png
...
0018_0000002391.png
0018_0000002393.png
...
```
