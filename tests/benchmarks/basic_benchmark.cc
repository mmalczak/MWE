#include <cstdlib>
#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include "src/cuda/matrix/include/matrix.hpp"
#include "src/cuda/matrix/include/matmul.hpp"

static void BM_BasicSquare(benchmark::State &state) {
  matrix::Matrix2 A;
  matrix::Matrix2 B;
  matrix::Matrix2 C;
  int32_t m = state.range(0);
  int32_t p = state.range(0);
  int32_t n = state.range(0);
  A.height = m;
  A.width = p;
  B.height = p;
  B.width = n;
  C.height = m;
  C.width = n;
  A.elements = (float *)malloc(m * p * sizeof(float));
  B.elements = (float *)malloc(p * n * sizeof(float));
  C.elements = (float *)malloc(m * n * sizeof(float));
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
  double seconds = 0;

  for (auto _ : state) {
    cudaEventRecord(start);
    matrix::MatMul(A, B, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    seconds = milliseconds / 1000;
    state.SetIterationTime(seconds);
  }
}

BENCHMARK(BM_BasicSquare)->RangeMultiplier(2)->Range(1 << 8, 1 << 13)->UseManualTime();

BENCHMARK_MAIN();
