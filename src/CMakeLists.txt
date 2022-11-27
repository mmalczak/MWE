set(HEADER_LIST "${mwe_SOURCE_DIR}/include/mwe/matrix.hpp")

add_library(matrix_library matrix.cpp ${HEADER_LIST})

target_include_directories(matrix_library PUBLIC ../include)

target_compile_features(matrix_library PUBLIC cxx_std_17)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST})

