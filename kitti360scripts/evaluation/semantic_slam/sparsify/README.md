This script downsamples a given point cloud.
It loops over all points and checks whether to keep or discard a point one by one. The output point cloud is first initialized to empty. If a point's distance to its nearest neighbor in the output point cloud is smaller than a given threshold, this point will be discarded. This downsampling ensures uniform downsampling without shifting the locations of the original points.

## Build

```
mkdir build && cd build
cmake ..
make
```

## Run

We provide an example script
```
./run.sh
```
to demonstrate the usage of our downsampling script.


