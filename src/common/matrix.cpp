#include "common/include/matrix.hpp"
#include "common/include/utils.hpp"

#include <cstdint>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

namespace matrix {

// We assume C matrix initialized with zeros
void multiply(int32_t M, int32_t P, int32_t N, float *A, float *B, float *C) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < P; k++) {
        C[i * N + j] += A[i * P + k] * B[k * N + j];
      }
    }
  }
}

} // namespace matrix
