cmake_minimum_required(VERSION 3.20)

project(
    mwe
    VERSION 0.1
    DESCRIPTION "Parallel math libraries"
    LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(tests)
