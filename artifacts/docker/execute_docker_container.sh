#!/usr/bin/env bash

set -e

NAME="plenoxel"

# Check if the container exists.
if ! docker ps -a --format "{{.Names}}" | grep -q "^$NAME$"; then
  echo "Container [$NAME] does not exist"
  exit 1
fi
# Check if the container is running.
if ! docker ps --format "{{.Names}}" | grep -q "^$NAME$"; then
  echo "Starting container [$NAME] ..."
  docker start $NAME >/dev/null
fi

docker exec -it \
  -u $(id -u -n) \
  -e USER \
  $NAME \
  /bin/bash
