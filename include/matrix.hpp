namespace matrix {

struct Matrix {
  unsigned int n_rows, n_columns;
  float *data;
};

Matrix *generateMatrix(int M, int N);
void printMatrix(int M, int N, float *data);
Matrix *multiplyMatrices(int M, int P, int N, float *matrix1, float *matrix2, float *matrixOut);

} // namespace matrix
