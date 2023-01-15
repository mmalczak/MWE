# MWE

## Code formatting
> ./scripts/code_format.sh

## Local environment
### Debug build
> cmake -B ${PWD}/build/ -D CMAKE_BUILD_TYPE="Debug" -G Ninja
> ninja -C ${PWD}/build all

### Release build
> cmake -B ${PWD}/build/ -D CMAKE_BUILD_TYPE="Release" -G Ninja
> ninja -C ${PWD}/build all

### Run example app
> ${PWD}/build/examples/basic_sample/basic_sample

### Run Unit Tests
> export BLOBS_DIR="${PWD}/tests/blobs/"
> cd build && ctest ${PWD}/build/ --verbose

## Docker container build
### Debug build
> docker build -t mwe --target=debug .

### Release build
> docker build -t mwe --target=release .

### Run example app
> docker run -it --rm --gpus all mwe /src/mwe/build/examples/basic_sample/basic_sample

### Run Unit Tests
> docker run -it --rm --gpus all mwe sh -c 'cd /src/mwe/build && ctest --verbose'

### Enter the container
> docker run -it mwe bash

Instruction to run on gpu:
1. Install NVIDIA driver for the GPU
2. Install CUDA toolkit
3. Install CUDA docker toolkit:
  - https://collabnix.com/introducing-new-docker-cli-api-support-for-nvidia-gpus-under-docker-engine-19-03-0-beta-release/
  - Remember to systemctl restart docker or reboot
