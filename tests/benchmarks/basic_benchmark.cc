#include "src/cuda/matrix/include/matrix.hpp"
#include "src/cuda/matrix/include/matmul.hpp"
#include "tests/common/include/blob_files.hpp"

#include <cstdlib>

#include <benchmark/benchmark.h>

matrix::Matrix construct_matrix(int32_t M, int32_t N) {
  float *p = static_cast<float *>(calloc(M * N, sizeof(float)));
  return matrix::Matrix{.M = M, .N = N, .data = p};
}

matrix::Matrix construct_matrix(int32_t M, int32_t N, std::string path) {
  matrix::Matrix m = construct_matrix(M, N);
  blob_files::load<float>(path, m.data);
  return m;
}

class BasicTest {
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

  matrix::Matrix a;
  matrix::Matrix b;
  matrix::Matrix c;
};

static void BM_BasicSquare(benchmark::State &state) {
  float A[4] = {1, 2, 3, 4};
  float B[4] = {1, 2, 3, 4};
  int32_t M = 2;
  int32_t P = 2;
  int32_t N = 2;
  float res[4] = {0, 0, 0, 0};
  for (auto _ : state)
    matrix::multiply(res, A, B, M, P, N);
}

BENCHMARK(BM_BasicSquare);

static void BM_WithBinaryFiles(benchmark::State &state) {
  BasicTest basicTest;
  std::string tc_name = "tc1";
  basicTest.load_data(tc_name);

  matrix::Matrix res = construct_matrix(basicTest.c.M, basicTest.c.N);
  for (auto _ : state)
    matrix::multiply(res.data, basicTest.a.data, basicTest.b.data, basicTest.a.M, basicTest.a.N, basicTest.b.N);

  basicTest.free_data();
  free(res.data);
}

BENCHMARK(BM_WithBinaryFiles);

BENCHMARK_MAIN();
