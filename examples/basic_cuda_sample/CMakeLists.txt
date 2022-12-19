add_executable(basic_cuda_sample "basic_cuda_sample.cpp")
target_link_libraries(basic_cuda_sample PRIVATE matrix_cuda_library utils_library)
set_target_properties(basic_cuda_sample
                    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(basic_cuda_sample PROPERTIES CUDA_ARCHITECTURES
                                                        "50")
