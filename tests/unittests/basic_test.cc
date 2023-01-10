#include "src/cuda/matrix/include/matrix.hpp"
#include "src/cuda/matrix/include/matmul.hpp"
#include "tests/common/include/matrix_test.hpp"

#include <cstdlib>

#include <gtest/gtest.h>

class MatrixUnittest : public ::testing::Test {
public:
  float tolerance = 1e-4;
  void verify(const matrix::Matrix &m1, const matrix::Matrix &m2) {
    ASSERT_EQ(m1.M, m2.M);
    ASSERT_EQ(m1.N, m2.N);

    for (int32_t i = 0; i < m1.M * m1.N; i++) {
      EXPECT_NEAR(m1.data[i], m2.data[i], tolerance);
    }
  }

  MatrixTest matrixTest;
};

TEST(basicSquare, returnsMatrixMulResult) {
  float A[4] = {1, 2, 3, 4};
  float B[4] = {1, 2, 3, 4};
  float C[4] = {7, 10, 15, 22};
  int32_t m = 2;
  int32_t p = 2;
  int32_t n = 2;
  float res[4] = {0, 0, 0, 0};

  matrix::multiply(res, A, B, m, p, n);

  for (int32_t i = 0; i < m * n; i++) {
    ASSERT_EQ(C[i], res[i]);
  }
}

TEST_F(MatrixUnittest, withBinaryFiles) {
  std::string tc_name = "tc1";
  matrixTest.load_data(tc_name);

  matrix::Matrix res = matrixTest.construct_matrix(matrixTest.c.M, matrixTest.c.N);
  matrix::multiply(res.data, matrixTest.a.data, matrixTest.b.data, matrixTest.a.M, matrixTest.a.N, matrixTest.b.N);

  verify(matrixTest.c, res);

  matrixTest.free_data();
  free(res.data);
}
