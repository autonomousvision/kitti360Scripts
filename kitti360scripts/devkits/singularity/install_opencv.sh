#!/bin/bash
set -e

opencv="opencv"

apt-get -y install build-essential
apt-get -y install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
apt-get -y install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev


if [ ! -d $opencv ]
then
# For compatibility purpose, use version 2.4 instead. 
# However, for version 3.0, it should also be ok.
	git clone -b 2.4 https://github.com/opencv/opencv.git
fi 

cd $opencv
mkdir -p release
cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..

make -j2
make install
