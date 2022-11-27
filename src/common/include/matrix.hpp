#include <cstdint>

namespace matrix {

struct Matrix {
  int32_t M, N;
  float *data;
};

Matrix *multiply(int32_t M, int32_t P, int32_t N, float *matrix1, float *matrix2, float *matrixOut);

} // namespace matrix
