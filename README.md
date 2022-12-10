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
> ${PWD}/build/examples/basic_sample/mwe

### Run Unit Tests
> export BLOBS_DIR="${PWD}/tests/blobs/"
> ctest --test-dir ${PWD}/build/ --verbose

## Docker container build
### Debug build
> docker build -t mwe --target=debug .

### Release build
> docker build -t mwe --target=release .

### Run example app
> docker run -it mwe /src/mwe/build/examples/basic_sample/mwe

### Run Unit Tests
> docker run -it mwe sh -c 'ctest --test-dir /src/mwe/build'

### Enter the container
> docker run -it mwe bash
