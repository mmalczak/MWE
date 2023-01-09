#include "src/cuda/matrix/include/matrix.hpp"
#include "src/cuda/matrix/include/matmul.hpp"
#include "tests/common/include/blob_files.hpp"

#include <cstdlib>

#include <gtest/gtest.h>

constexpr float tolerance = 1e-4;

matrix::Matrix construct_matrix(int32_t M, int32_t N) {
  float *p = static_cast<float *>(calloc(M * N, sizeof(float)));
  return matrix::Matrix{.M = M, .N = N, .data = p};
}

matrix::Matrix construct_matrix(int32_t M, int32_t N, std::string path) {
  matrix::Matrix m = construct_matrix(M, N);
  blob_files::load<float>(path, m.data);
  return m;
}

class BasicTest : public ::testing::Test {
public:
  void load_data(std::string tc_name) {
    std::string a_path = tc_name + "/" + "A";
    std::string b_path = tc_name + "/" + "B";
    std::string c_path = tc_name + "/" + "C";
    std::string dim_path = tc_name + "/" + "Dims";

    int32_t dims[3];
    blob_files::load(dim_path, dims);

    int32_t M = dims[0];
    int32_t P = dims[1];
    int32_t N = dims[2];

    a = construct_matrix(M, P, a_path);
    b = construct_matrix(P, N, b_path);
    c = construct_matrix(M, N, c_path);
  }

  void free_data() {
    free(a.data);
    free(b.data);
    free(c.data);
  }

  void verify(const matrix::Matrix &m1, const matrix::Matrix &m2) {
    ASSERT_EQ(m1.M, m2.M);
    ASSERT_EQ(m1.N, m2.N);

    for (int32_t i = 0; i < m1.M * m1.N; i++) {
      EXPECT_NEAR(m1.data[i], m2.data[i], tolerance);
    }
  }

  matrix::Matrix a;
  matrix::Matrix b;
  matrix::Matrix c;
};

TEST_F(BasicTest, basicSquare) {
  float A[4] = {1, 2, 3, 4};
  float B[4] = {1, 2, 3, 4};
  float C[4] = {7, 10, 15, 22};
  int32_t M = 2;
  int32_t P = 2;
  int32_t N = 2;
  float res[4] = {0, 0, 0, 0};

  matrix::multiply(res, A, B, M, P, N);

  for (int32_t i = 0; i < M * N; i++) {
    ASSERT_EQ(C[i], res[i]);
  }
}

TEST_F(BasicTest, withBinaryFiles) {
  std::string tc_name = "tc1";
  load_data(tc_name);

  matrix::Matrix res = construct_matrix(c.M, c.N);
  matrix::multiply(res.data, a.data, b.data, a.M, a.N, b.N);

  verify(c, res);

  free_data();
  free(res.data);
}
