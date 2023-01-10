set(srcs ${CMAKE_CURRENT_SOURCE_DIR}/basic_benchmark.cc)

add_executable(matmul_benchmark ${srcs})

set_target_properties(matmul_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(matmul_benchmark PROPERTIES CUDA_ARCHITECTURES "50")
target_link_libraries(matmul_benchmark benchmark::benchmark pthread
                      matrix_library matrix_test_library)
