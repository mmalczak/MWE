#include "common/include/matrix.hpp"
#include "common/include/utils.hpp"

#include <cstdlib>

#include <gtest/gtest.h>

constexpr float tolerance = 1e-4;

class BasicTest : public ::testing::Test {
public:
  matrix::Matrix construct_matrix(int32_t M, int32_t N) {
    matrix::Matrix a;

    a.data = static_cast<float *>(calloc(M * N, sizeof(float)));
    a.M = M;
    a.N = N;

    return a;
  }

  void verify(const matrix::Matrix &m1, const matrix::Matrix &m2) {
    ASSERT_EQ(m1.M, m2.M);
    ASSERT_EQ(m1.N, m2.N);

    int32_t M = m1.M;
    int32_t N = m1.N;

    for (int32_t i = 0; i < M; i++) {
      for (int32_t j = 0; j < N; j++) {
        EXPECT_NEAR(m1.data[i * M + j], m2.data[i * M + j], tolerance);
      }
    }
  }

  void load_data(std::string tc_name) {
    std::string a_path = tc_name + "/" + "A";
    std::string b_path = tc_name + "/" + "B";
    std::string c_path = tc_name + "/" + "C";
    std::string dim_path = tc_name + "/" + "Dims";

    std::vector<int32_t> dims;
    utils::read_blob_file(dim_path, dims);

    int32_t M = dims[0];
    int32_t P = dims[1];
    int32_t N = dims[2];

    a = construct_matrix(M, P);
    b = construct_matrix(P, N);
    c = construct_matrix(M, N);

    utils::load<float>(a_path, a.data);
    utils::load<float>(b_path, b.data);
    utils::load<float>(c_path, c.data);
  }

  void free_data() {
    free(a.data);
    free(b.data);
    free(c.data);
  }

  matrix::Matrix a;
  matrix::Matrix b;
  matrix::Matrix c;
};

TEST_F(BasicTest, testWithBinaryFiles) {
  std::string tc_name = "tc1";

  load_data(tc_name);

  matrix::Matrix res = construct_matrix(c.M, c.N);
  matrix::multiply(a.M, a.N, b.N, a.data, b.data, res.data);

  verify(c, res);

  free_data();
  free(res.data);
}
