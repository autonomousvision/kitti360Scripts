## Accumulation of raw 3D scans

This folder contains code to accumulate raw 3D velodyne/sick scans.

### Dependencies

* __OpenCV__ 2.4:
Install according to the following doc:
http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html

* __Eigen__ 3.3.1:

```
sudo apt install libeigen3-dev
```

It is recommended to use a singularity container. The recipe can be found in the ```../singularity/``` folder. You can build a container by 

```
sudo singularity build accumuLaser.simg accumuLaser.recipe
```


### Installation

```
mkdir build && cd build
cmake ..
make -j4
```


### Accumulation

#### Input preparation

Please make sure that you have downloaded the following folders from our website and they are located in ```KITTI360_DATASET```.

 * calibrations
 * data_3d_raw
 * data_poses

Please check if you have downloaded the __data_3d_timestamps.zip__ from the updated download script. It contains timestamps of our raw 3D scans which are required for accumulation.

#### Run

You can find scripts to run the accumulation under the ```../scripts``` folder.

```
$cd ../scripts/
$./run_accumulation.sh $KITTI360_DATASET $OUTPUT_DIR $SEQUENCE $FIRST_FRAME $LAST_FRAME $DATA_SOURCE
```

Here ```DATA_SOURCE``` specifies the type of laser scans to accumulate:

 * 0: accumulate sick scans only;
 * 1: accumulate velodyne scans only;
 * 2: accumulate both sick and velodyne scans.

Here is an example:
```
./run_accumulation.sh ./KITTI-360 output 2013_05_28_drive_0003_sync 2 282 2
```

Here is the example if you use the singularity container:
```
singularity exec ../singularity/accumuLaser.simg  ./run_accumulation.sh ./KITTI-360 output 2013_05_28_drive_0003_sync 2 282 2
```


#### Output

The output will be saved in ```$OUTPUT_DIR/$SEQUENCE_${FIRST_FRAME}_${LAST_FRAME}```, including the following files:

  * lidar_points_{type}.dat: Each row contains a 3D point in the format of ```x y z r g b```, where the color is generated according to the height, i.e., the ```z``` axis.
  * lidar_timestamp_{type}.dat: Each row contains the timestamp of the 3D point specified in the same row of lidar_points_{type}.dat.
  * lidar_loc.dat: Each row contains the system pose when the 3D point specified in the same row of lidar_points_{type}.dat is captured.
  * lidar_pose.dat: This contains the system poses from ```FIRST_FRAME``` to ```LAST_FRAME```.

Here ```type``` can be _all_, _velodyne_ or _sick_ depending on the ```DATA_SOURCE``` argument.
