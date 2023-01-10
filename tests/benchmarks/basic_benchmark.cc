#include "src/cuda/matrix/include/matrix.hpp"
#include "src/cuda/matrix/include/matmul.hpp"
#include "tests/common/include/matrix_test.hpp"

#include <cstdlib>

#include <benchmark/benchmark.h>

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
  MatrixTest basicTest;
  std::string tc_name = "tc1";
  basicTest.load_data(tc_name);

  matrix::Matrix res = basicTest.construct_matrix(basicTest.c.M, basicTest.c.N);
  for (auto _ : state)
    matrix::multiply(res.data, basicTest.a.data, basicTest.b.data, basicTest.a.M, basicTest.a.N, basicTest.b.N);

  basicTest.free_data();
  free(res.data);
}

BENCHMARK(BM_WithBinaryFiles);

BENCHMARK_MAIN();
