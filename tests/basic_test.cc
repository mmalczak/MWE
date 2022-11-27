#include "common/include/matrix.hpp"

#include <gtest/gtest.h>

constexpr int32_t max_rows = 32;
constexpr int32_t max_cols = 32;

class BasicTest : public ::testing::Test {
public:
  float m1_data[max_rows][max_cols]{};
  float m2_data[max_rows][max_cols]{};
  float exp_out[max_rows][max_cols]{};
};

TEST_F(BasicTest, test_2x2) {
  constexpr int32_t rows = 2;
  constexpr int32_t columns = 2;

  matrix::Matrix m1{.n_rows = rows, .n_columns = columns, .data = &m1_data[0][0]};
  matrix::Matrix m2{.n_rows = rows, .n_columns = columns, .data = &m2_data[0][0]};

  float out_data[max_rows][max_cols]{};
  matrix::Matrix out{.n_rows = rows, .n_columns = columns, .data = &out_data[0][0]};

  matrix::multiply(m1.n_rows, m1.n_columns, m2.n_columns, m1.data, m2.data, out.data);

  for (int32_t i = 0; i < rows; i++) {
    for (int32_t k = 0; k < columns; k++) {
      EXPECT_NEAR(out.data[i * columns + k], exp_out[i][k], 1e-4f);
    }
  }
}
