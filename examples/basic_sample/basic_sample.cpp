#include <cstdint>
#include "src/cuda/matrix/include/matmul.hpp"
#include "src/common/include/utils.hpp"

int main(int argc, char *argv[]) {
  void *A, *B, *C;
  int32_t m = 1024;
  int32_t n = 1024;
  int32_t p = 1024;
  A = malloc(m*p*sizeof(float));
  B = malloc(p*n*sizeof(float));
  C = malloc(m*n*sizeof(float));
  for(int i=0;i<100;i++) {
    matrix::multiply(static_cast<float *>(C), static_cast<float *>(A), static_cast<float *>(B), m, p, n);
    matrix::multiply_optimized(static_cast<float *>(C), static_cast<float *>(A), static_cast<float *>(B), m, p, n);
  }
}
