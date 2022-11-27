set(srcs ${CMAKE_CURRENT_SOURCE_DIR}/basic_test.cc)

add_executable(matmul_test ${srcs})

target_link_libraries(matmul_test ${GTEST_LIBRARIES} pthread gtest_main
                      matrix_library utils_library)

if(NOT ("${GTEST_DEPENDENCY}" STREQUAL ""))
  add_dependencies(matmul_test ${GTEST_DEPENDENCY})
endif()

gtest_discover_tests(matmul_test ${srcs} TEST_PREFIX matmul_test)
