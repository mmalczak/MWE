add_library(matrix_cuda_library STATIC ${CMAKE_CURRENT_SOURCE_DIR}/matrix_cuda.cu)
set_target_properties(matrix_cuda_library
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(matrix_cuda_library PROPERTIES CUDA_ARCHITECTURES
                                                          "50")
target_include_directories(
      matrix_cuda_library PRIVATE ${PROJECT_SOURCE_DIR}/src/cuda/Common/ 
                                  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
