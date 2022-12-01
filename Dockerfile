#################################################################
FROM ubuntu:22.04 as build_env
WORKDIR /src/
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y install --no-install-recommends build-essential clang cmake git libgtest-dev pkg-config python3-pip vim meson

COPY . /src/mwe/
RUN cp /src/mwe/common_commands /root/.bash_history
RUN rm -rf build

ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++
ENV BLOBS_DIR="/src/mwe/tests/blobs/"

#################################################################
FROM build_env as debug
WORKDIR /src/mwe/
RUN cmake -B ${PWD}/build/ -D CMAKE_BUILD_TYPE="Debug" -G Ninja
RUN ninja -C ./build all

#################################################################
FROM build_env as release
WORKDIR /src/mwe/
RUN cmake -B ${PWD}/build/ -D CMAKE_BUILD_TYPE="Release" -G Ninja
RUN ninja -C ./build all

