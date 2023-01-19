#include <cstdlib>
#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include "src/cuda/matrix/include/matrix.hpp"
#include "src/cuda/matrix/include/matmul.hpp"

static void BM_BasicSquare(benchmark::State &state) {
  void *A, *B, *C;
  int32_t m = state.range(0);
  int32_t p = state.range(0);
  int32_t n = state.range(0);
  A = malloc(m * p * sizeof(float));
  B = malloc(p * n * sizeof(float));
  C = malloc(m * n * sizeof(float));
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
  double seconds = 0;

  for (auto _ : state) {
    cudaEventRecord(start);
    matrix::multiply(static_cast<float *>(C), static_cast<float *>(A), static_cast<float *>(B), m, p, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    seconds = milliseconds / 1000;
    state.SetIterationTime(seconds);
  }
}

BENCHMARK(BM_BasicSquare)->RangeMultiplier(2)->Range(1 << 8, 1 << 13)->UseManualTime();

static void BM_BasicSquareKernel(benchmark::State &state) {
  void *devA, *devB, *devC;
  int32_t m = state.range(0);
  int32_t p = state.range(0);
  int32_t n = state.range(0);
  cudaMalloc(&devA, m * p * sizeof(float));
  cudaMalloc(&devB, p * n * sizeof(float));
  cudaMalloc(&devC, m * n * sizeof(float));
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
  double seconds = 0;
  for (auto _ : state) {
    cudaEventRecord(start);
    matrix::run_multiply_kernel(static_cast<float *>(devC), static_cast<float *>(devA), static_cast<float *>(devB), m, p, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    seconds = milliseconds / 1000;
    state.SetIterationTime(seconds);
  }
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);
}

BENCHMARK(BM_BasicSquareKernel)->RangeMultiplier(2)->Range(1 << 8, 1 << 13)->UseManualTime();

BENCHMARK_MAIN();
