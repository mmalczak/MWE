add_executable(mwe mwe.cpp)
target_compile_features(mwe PRIVATE cxx_std_17)

target_link_libraries(mwe PRIVATE matrix_library)
