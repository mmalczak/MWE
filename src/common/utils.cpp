#include "src/common/include/utils.hpp"

#include <cstdint>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>

namespace utils {

void print(int32_t M, int32_t N, float *data) {
  for (int32_t j = 0; j < M; j++) {
    for (int32_t i = 0; i < N; i++) {
      printf("%f ", *data++);
    }
    printf("\n");
  }
  printf("\n");
}
} // namespace utils
