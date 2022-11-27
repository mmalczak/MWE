namespace matrix {

struct Matrix {
  unsigned int n_rows, n_columns;
  float *data;
};

Matrix *multiply(int M, int P, int N, float *matrix1, float *matrix2, float *matrixOut);

} // namespace matrix
