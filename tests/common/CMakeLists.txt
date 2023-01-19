add_library(blob_files_library STATIC
            ${CMAKE_CURRENT_SOURCE_DIR}/blob_files.cpp)
add_library(matrix_test_library STATIC
            ${CMAKE_CURRENT_SOURCE_DIR}/matrix_test.cpp)
target_link_libraries(matrix_test_library blob_files_library)
