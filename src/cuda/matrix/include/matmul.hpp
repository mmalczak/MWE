#include <cstdint>

namespace matrix {
typedef struct {
  int width;
  int height;
  int stride;
  float *elements;
} Matrix2;

// Thread block size
#define BLOCK_SIZE 16
void multiply(float *C, float *A, float *B, int32_t m, int32_t p, int32_t n);
void run_multiply_kernel(float *devC, float *devA, float *devB, int32_t m, int32_t p, int32_t n);
void MatMul(const Matrix2 A, const Matrix2 B, Matrix2 C);
} // namespace matrix
