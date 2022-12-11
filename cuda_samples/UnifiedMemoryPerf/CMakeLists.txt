add_executable(UnifiedMemoryPerf_sample commonKernels.cu helperFunctions.cpp
                                        matrixMultiplyPerf.cu)
set_target_properties(UnifiedMemoryPerf_sample
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(UnifiedMemoryPerf_sample PROPERTIES CUDA_ARCHITECTURES
                                                          "86")
target_include_directories(
  UnifiedMemoryPerf_sample PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                                   ${PROJECT_SOURCE_DIR}/cuda_samples/Common/)
