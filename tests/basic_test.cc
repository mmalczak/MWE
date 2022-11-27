#include "common/include/matrix.hpp"

#include <gtest/gtest.h>

constexpr int32_t max_rows = 32;
constexpr int32_t max_cols = 32;
constexpr float tolerance = 1e-6;

class BasicTest : public ::testing::Test {
public:
  matrix::Matrix create_matrix(int32_t M, int32_t N, float *data) { return matrix::Matrix{.M = M, .N = N, .data = data}; }

  void verify(const matrix::Matrix &m1, const matrix::Matrix &m2) {
    EXPECT_EQ(m1.M, m2.N);
    EXPECT_EQ(m1.N, m2.N);

    int32_t M = m1.M;
    int32_t N = m1.N;

    for (int32_t i = 0; i < M; i++) {
      for (int32_t j = 0; j < N; j++) {
        EXPECT_NEAR(m1.data[i * M + j], m2.data[i * M + j], tolerance);
      }
    }
  }
};

TEST_F(BasicTest, test_2x2) {
  constexpr int32_t M = 2;
  constexpr int32_t N = 2;

  float m1_data[max_rows * max_cols]{};
  float m2_data[max_rows * max_cols]{};
  float out_data[max_rows * max_cols]{};
  float exp_data[max_rows * max_cols]{};

  matrix::Matrix m1 = create_matrix(M, N, &m1_data[0]);
  matrix::Matrix m2 = create_matrix(M, N, &m2_data[0]);
  matrix::Matrix out = create_matrix(M, N, &out_data[0]);
  matrix::Matrix exp = create_matrix(M, N, &exp_data[0]);

  matrix::multiply(m1.M, m1.N, m2.N, m1.data, m2.data, out.data);

  verify(out, exp);
}
