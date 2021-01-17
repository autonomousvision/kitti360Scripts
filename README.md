# The KITTI-360 Dataset

This repository contains scripts for inspection of the KITTI-360 dataset. This large-scale dataset contains 320k images and 100k laser scans in a driving distance of 73.7km.  We annotate both static and dynamic 3D scene elements with rough bounding primitives and transfer this information into the image domain, resulting in dense semantic & instance annotations on both 3D point clouds and 2D images.

Details and download are available at: www.cvlibs.net/datasets/kitti-360


## Dataset Structure

Dataset structure and data formats are available at: www.cvlibs.net/datasets/kitti-360/documentation.php

## Scripts

### Installation

Install `kitti360Scripts` with `pip`
```
pip install git+https://github.com/autonomousvision/kitti360Scripts.git
```

For the 2D graphical tools you additionally need to install
```
sudo apt install python-tk python-qt5
```

We use open3D to visualize 3D point clouds and 3D bounding boxes:
```
pip install open3d
```

### Usage

This scripts contains helpers for loading and visualizing our dataset. For inspection, please download the dataset and add the root directory to your system path at first:

```
export KITTI360_DATASET=/PATH/TO/THE/DATASET
```

You can inspect the 2D images and labels using the following tool:
```
cd kitti360scripts/viewer
python kitti360Viewer.py
```

You can visualize the 3D fused point clouds and labels using the following tool:
```
cd kitti360scripts/viewer
python kitti360Viewer3D.py -sequence ${sequence}
```

### Package Content

The package is structured as follows
 - `helpers`: helper files that are included by other scripts
 - `viewer`: view the 2D image & labels and the 3D fused point clouds & labels

Note that all files have a small documentation at the top. Most important files
 - `helpers/labels.py`: central file defining the IDs of all semantic classes and providing mapping between various class properties.
 - `helpers/annotations.py`: central file containing the loaders for all labels including 3D bounding boxes and fused point clouds


## Acknowledgment

The 2D graphical tool is adapted from Cityscapes. 

## Contact

Please feel free to contact us with any questions, suggestions or comments:

* Yiyi Liao, Andreas Geiger 
* yiyi.liao@tue.mpg.de, a.geiger@uni-tuebingen.de
* www.cvlibs.net/datasets/kitti-360 
