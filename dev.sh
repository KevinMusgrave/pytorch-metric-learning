#!/usr/bin/env bash

CONT_NAME=lj-dev-container-gpu-metrics

if [ -z "$CONT_VERSION" ]
then
  CONT_VERSION=latest
fi

if [ -z "$BASE" ]
then
  BASE=torch
fi

# DETERMINE IF GPU IS AVAILABLE
if [ -z "$USE_GPU" ]
then
  echo "Using CPU"
  GPU=""
  GPU_FLAGS=""
else
  echo "Using GPU"
  GPU="-gpu" # "" or "gpu"
  GPU_FLAGS="--gpus all"
fi

export GROUP_ID=1004

# DETERMINE IF GPU IS AVAILABLE
echo "Using ${BASE} ${GPU} ${NETWORK}"

dev_build() {
  echo "Building ${CONT_NAME}:${CONT_VERSION}"
  docker build \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    -f Dockerfile \
    -t ${CONT_NAME}:${CONT_VERSION} .
  echo "Done building ${CONT_NAME}:${CONT_VERSION}"
}

NETWORK="--network host"

dev_bash() {
  echo "Starting ${CONT_NAME}:${CONT_VERSION}"
  docker run ${GPU_FLAGS} \
    --mount type=bind,source="${LAYERJOT_HOME}",target=/layerjot \
    --mount type=bind,source="${LAYERJOT_MODELS}",target=/models \
    --mount type=bind,source="${LJ_DATA}",target=/data \
    --shm-size 8G \
    --rm ${NETWORK} -it ${CONT_NAME}:${CONT_VERSION} bash	
}
