cmake_minimum_required(VERSION 3.16)

set(PACKAGE_NAME "MWE_matmul")
set(PACKAGE_VERSION "0.1")

include("CMakeUtils/compiler_config.cmake")

project(
  ${PACKAGE_NAME}
  VERSION ${PACKAGE_VERSION}
  DESCRIPTION "Parallel math libraries"
  LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type defined, defaulting to 'Debug'")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include_directories("${PROJECT_SOURCE_DIR}/src/")

find_package(GTest)
include_directories(${GTEST_INCLUDE_DIRS})
enable_testing()

add_subdirectory(examples)
add_subdirectory(src)
add_subdirectory(tests)
