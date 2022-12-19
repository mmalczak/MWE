#include "cuda/include/matrix_cuda.hpp"
#include "common/include/utils.hpp"

int main(int argc, char *argv[]) {
  float A[] = {1, 2, 3, 4};
  float B[] = {1, 2, 3, 4};
  float C[] = {0, 0, 0, 0};
  int32_t M = 2;
  int32_t P = 2;
  int32_t N = 2;

  matrix_cuda::multiply_matrix(C, A, B, M, P, N);
  utils::print(M, N, C);
}
