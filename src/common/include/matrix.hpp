#include <cstdint>

namespace matrix {

struct Matrix {
  int32_t n_rows, n_columns;
  float *data;
};

Matrix *multiply(int32_t M, int32_t P, int32_t N, float *matrix1, float *matrix2, float *matrixOut);

} // namespace matrix
