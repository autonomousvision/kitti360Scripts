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

## Reference

If you find this code or our dataset helpful in your research, please use the following BibTeX entry.

```
@article{Liao2021ARXIV, 
   title   = {{KITTI}-360: A Novel Dataset and Benchmarks for Urban Scene Understanding in 2D and 3D}, 
   author  = {Yiyi Liao and Jun Xie and Andreas Geiger}, 
   journal = {arXiv preprint arXiv:2109.13410},
   year    = {2021}, 
}
```

## Contact

Please feel free to contact us with any questions, suggestions or comments:

* Yiyi Liao, Andreas Geiger 
* yiyi.liao@tue.mpg.de, a.geiger@uni-tuebingen.de
* www.cvlibs.net/datasets/kitti-360 

## License

Our __utility scripts__ in this repository are released under the following MIT license. 

---

MIT License

Copyright (c) 2021 Autonomous Vision Group

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

Our __datasets__ and __benchmarks__ are copyright by us and published under the [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License](https://creativecommons.org/licenses/by-nc-sa/3.0/). This means that you must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license.

