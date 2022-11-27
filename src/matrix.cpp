#include "matrix.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

namespace matrix {

Matrix *generateMatrix(int M, int N) {
  int size = M * N;
  srand(time(NULL));
  Matrix *matrixObj = (Matrix *)malloc(sizeof(Matrix));
  if (matrixObj == NULL) {
    return NULL;
  }
  matrixObj->data = (float *)malloc(size * sizeof(float));
  if (matrixObj->data == NULL) {
    return NULL;
  }
  for (int i = 0; i < size; i++) {
    matrixObj->data[i] = rand();
  }
  return matrixObj;
}

void printMatrix(int M, int N, float *data) {
  for (int j = 0; j < M; j++) {
    for (int i = 0; i < N; i++) {
      int idx = i + j * N;
      printf("%f ", data[idx]);
    }
    printf("\n");
  }
  printf("\n");
}

Matrix *multiplyMatrices(int M, int P, int N, float *matrix1, float *matrix2, float *matrixOut) { return nullptr; }

} // namespace matrix
