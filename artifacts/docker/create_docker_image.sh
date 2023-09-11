#!/bin/bash

# Directory that contains current script
DIR=$(dirname $(realpath "$BASH_SOURCE"))

docker image build \
    --tag plenoxel \
    --build-arg usr=$(id -u -n) \
    --build-arg uid=$(id -u) \
    --build-arg grp=$(id -g -n) \
    --build-arg gid=$(id -g) \
    --file $DIR/dev.dockerfile $DIR
