#!/usr/bin/env python
#
# Enable cython support for slightly faster eval scripts:
# python -m pip install cython numpy
# CYTHONIZE_EVAL= python setup.py build_ext --inplace
#
# For MacOS X you may have to export the numpy headers in CFLAGS
# export CFLAGS="-I /usr/local/lib/python3.6/site-packages/numpy/core/include $CFLAGS"

import os
from setuptools import setup, find_packages

include_dirs = []
ext_modules = []
if 'CYTHONIZE_EVAL' in os.environ:
    from Cython.Build import cythonize
    import numpy as np
    include_dirs = [np.get_include()]

    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"

    pyxFile = os.path.join("kitti360scripts", "evaluation", "addToConfusionMatrix.pyx")
    ext_modules = cythonize(pyxFile)

with open("README.md") as f:
    readme = f.read()

config = {
    'name': 'kitti360Scripts',
    'description': 'Scripts for the KITTI-360 Dataset',
    'long_description': readme,
    'long_description_content_type': "text/markdown",
    'author': 'Yiyi Liao',
    'url': 'https://github.com/autonomousvision/kitti360Scripts',
    'license': 'https://github.com/autonomousvision/kitti360Scripts/blob/master/license.txt',
    'version': '1.0.0',
    'install_requires': ['numpy', 'matplotlib', 'pillow', 'pyyaml', 'scikit-image'],
    'setup_requires': ['setuptools>=18.0'],
    'packages': find_packages(),
    'scripts': [],
    'entry_points': {'gui_scripts': ['kitti360Viewer = kitti360scripts.viewer.kitti360Viewer:main']},
    'package_data': {'': ['icons/*.png']},
    'ext_modules': ext_modules,
    'include_dirs': include_dirs
}

setup(**config)
