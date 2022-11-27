#include "common/include/matrix.hpp"
#include "common/include/utils.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

namespace utils {

void print(int M, int N, float *data) {
  for (int j = 0; j < M; j++) {
    for (int i = 0; i < N; i++) {
      int idx = i + j * N;
      printf("%f ", data[idx]);
    }
    printf("\n");
  }
  printf("\n");
}

} // namespace utils
