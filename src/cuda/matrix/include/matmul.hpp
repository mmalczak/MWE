#include <cstdint>

namespace matrix {
void multiply(float *C, float *A, float *B, int32_t m, int32_t p, int32_t n);
void run_multiply_kernel(float *devC, float *devA, float *devB, int32_t m, int32_t p, int32_t n);

void multiply_optimized(float *C, float *A, float *B, int32_t m, int32_t p, int32_t n);
void run_multiply_kernel_optimized(float *devC, float *devA, float *devB, int32_t m, int32_t p, int32_t n);
} // namespace matrix
