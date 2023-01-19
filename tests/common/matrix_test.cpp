#include "src/cuda/matrix/include/matrix.hpp"
#include "tests/common/include/blob_files.hpp"
#include "tests/common/include/matrix_test.hpp"

#include <cstdlib>

matrix::Matrix MatrixTest::construct_matrix(int32_t M, int32_t N) {
  float *p = static_cast<float *>(calloc(M * N, sizeof(float)));
  return matrix::Matrix{.M = M, .N = N, .data = p};
}

matrix::Matrix MatrixTest::construct_matrix(int32_t M, int32_t N, std::string path) {
  matrix::Matrix m = construct_matrix(M, N);
  blob_files::load<float>(path, m.data);
  return m;
}

void MatrixTest::load_data(std::string tc_name) {
  std::string a_path = tc_name + "/" + "A";
  std::string b_path = tc_name + "/" + "B";
  std::string c_path = tc_name + "/" + "C";
  std::string dim_path = tc_name + "/" + "Dims";

  int32_t dims[3];
  blob_files::load(dim_path, dims);

  int32_t m = dims[0];
  int32_t p = dims[1];
  int32_t n = dims[2];

  a = construct_matrix(m, p, a_path);
  b = construct_matrix(p, n, b_path);
  c = construct_matrix(m, n, c_path);
}

void MatrixTest::free_data() {
  free(a.data);
  free(b.data);
  free(c.data);
}
