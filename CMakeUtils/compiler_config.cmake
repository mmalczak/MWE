set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(SPECIAL_GTEST_FLAG " -Wno-deprecated-declarations")

set(COMMON_CXX_FLAGS
    "${CXX_FLAGS} ${CMAKE_CXX_FLAGS} ${SPECIAL_GTEST_FLAG} -Wformat=2 -Wstrict-aliasing -Werror -Wpedantic -march=native -mtune=native -ffast-math -fno-builtin -fno-rtti -fno-exceptions"
)
set(ASAN_CXX_FLAGS
    " -fsanitize=address,undefined -fsanitize-undefined-trap-on-error")
set(DEBUG_CXX_FLAGS
    " -DDEBUG -DMALLOC_CHECK_=3 -O0 -fstack-protector-all -ggdb -Wunused")
set(RELEASE_CXX_FLAGS " -O2")

set(CMAKE_CXX_FLAGS_RELEASE ${COMMON_CXX_FLAGS}${RELEASE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG
    ${COMMON_CXX_FLAGS}${DEBUG_CXX_FLAGS}${ASAN_CXX_FLAGS})
