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
> ./build/apps/mwe

## Run Unit Tests
> cd build && ctest --verbose
