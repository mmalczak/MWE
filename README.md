# MWE

## Code formatting
> ./scripts/code_format.sh

## Build

### Debug
> cmake -B ${PWD}/build/ -D CMAKE_BUILD_TYPE="Debug" -G Ninja
> ninja -C ./build all

### Release
> cmake -B ${PWD}/build/ -D CMAKE_BUILD_TYPE="Release" -G Ninja
> ninja -C ./build all

## Run app
> ./build/examples/mwe

## Run Unit Tests
> export BLOBS_DIR="${PWD}/tests/blobs/"
> cd build && ctest --verbose
