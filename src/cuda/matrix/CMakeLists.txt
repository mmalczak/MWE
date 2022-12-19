add_library(matrix_library STATIC ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cu)
set_target_properties(matrix_library
                PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(matrix_library PROPERTIES CUDA_ARCHITECTURES "50")
target_include_directories(
                matrix_library PRIVATE ${PROJECT_SOURCE_DIR}/src/cuda/Common/
                ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
