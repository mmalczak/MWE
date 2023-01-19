#include <cuda.h>
#include <cuda_runtime_api.h>
#include "src/cuda/Common/helper_cuda.h"
#include <cstdint>

namespace matrix {

__global__ static void multiply_kernel(float *C, float *A, float *B, int32_t m, int32_t p, int32_t n) {
  int32_t i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < m) {
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
      int32_t k;
      float s = 0;
      for (k = 0; k < p; k++) {
        s += A[i * p + k] * B[k * n + j];
      }
      C[i * n + j] = s;
    }
  }
}

void run_multiply_kernel(float *devC, float *devA, float *devB, int32_t m, int32_t p, int32_t n) {
  int32_t K = 32;
  dim3 dimBlock(K, K);
  dim3 dimGrid((n + K - 1) / K, (m + K - 1) / K);
  matrix::multiply_kernel<<<dimGrid, dimBlock>>>(devC, devA, devB, m, p, n);
}

void multiply(float *C, float *A, float *B, int32_t m, int32_t p, int32_t n) {
  void *devA, *devB, *devC;
  checkCudaErrors(cudaSetDevice(0));
  checkCudaErrors(cudaMalloc(&devA, m * p * sizeof(float)));
  checkCudaErrors(cudaMalloc(&devB, p * n * sizeof(float)));
  checkCudaErrors(cudaMalloc(&devC, m * n * sizeof(float)));

  checkCudaErrors(cudaMemcpy(devA, A, m * p * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devB, B, p * n * sizeof(float), cudaMemcpyHostToDevice));

  run_multiply_kernel(static_cast<float *>(devC), static_cast<float *>(devA), static_cast<float *>(devB), m, p, n);

  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(C, devC, m * n * sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(devA));
  checkCudaErrors(cudaFree(devB));
  checkCudaErrors(cudaFree(devC));
}

} // namespace matrix
