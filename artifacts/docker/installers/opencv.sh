#!/usr/bin/env bash

set -e

INSTALL_PREFIX=/usr/local
if [[ ! -z $1 ]]; then
  INSTALL_PREFIX=$1
fi

OPENCV_VERSION=4.5.3

mkdir -p /tmp/installers
rm -rf /tmp/installers/*
pushd /tmp/installers

# Check dependencies:
# https://docs.opencv.org/4.5.3/d7/d9f/tutorial_linux_install.html
wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
unzip ${OPENCV_VERSION}.zip
rm ${OPENCV_VERSION}.zip

wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip
unzip ${OPENCV_VERSION}.zip
rm ${OPENCV_VERSION}.zip

pushd opencv-${OPENCV_VERSION}
mkdir build
pushd build
# /gaussian-splatting/cmake/cmake-3.27.4-linux-x86_64/bin/cmake
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
  -D ENABLE_PRECOMPILED_HEADERS=OFF -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules \
  -D WITH_VTK=ON -D WITH_CUDA=OFF -D OPENCV_FORCE_3RDPARTY_BUILD=ON -D WITH_PROTOBUF=OFF -D BUILD_PROTOBUF=OFF \
  -D BUILD_LIST=calib3d,highgui,imgcodecs,flann,features2d,gapi,video,videoio,imgproc,line_descriptor ..
make -j$(($(nproc)-1))
make install
popd
popd

rm -rf opencv-${OPENCV_VERSION} opencv_contrib-${OPENCV_VERSION}

popd
