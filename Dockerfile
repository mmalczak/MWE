#################################################################
FROM ubuntu:22.04 as build_env
WORKDIR /src/
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y install --no-install-recommends build-essential clang cmake git libgtest-dev pkg-config python3-pip vim meson wget

RUN wget --no-check-certificate https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb && rm cuda-keyring_1.0-1_all.deb
RUN apt-get -y update && apt-get -y install --no-install-recommends cuda

COPY . /src/mwe/
RUN rm -rf /src/mwe/build/
RUN cp /src/mwe/common_commands /root/.bash_history

ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang
ENV CUDACXX=/usr/local/cuda/bin/nvcc
ENV BLOBS_DIR="/src/mwe/tests/blobs/"

#################################################################
FROM build_env as debug
WORKDIR /src/mwe/

ENV ASAN_OPTIONS=protect_shadow_gap=0

RUN cmake -B ${PWD}/build/ -D CMAKE_BUILD_TYPE="Debug" -G Ninja
RUN ninja -C ./build/ all

#################################################################
FROM build_env as release
WORKDIR /src/mwe/
RUN cmake -B ${PWD}/build/ -D CMAKE_BUILD_TYPE="Release" -G Ninja
RUN ninja -C ./build all

