#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

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

void multiply(float *C, float *A, float *B, int32_t m, int32_t p, int32_t n) {
  float *devA, *devB, *devC;
  int32_t N = 32;
  int32_t M = 32;
  checkCudaErrors(cudaSetDevice(0));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&devA), m * p * sizeof(float)));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&devB), p * n * sizeof(float)));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&devC), m * n * sizeof(float)));

  checkCudaErrors(cudaMemcpy(devA, A, m * p * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devB, B, p * n * sizeof(float), cudaMemcpyHostToDevice));

  dim3 dimBlock(N, M);
  dim3 dimGrid((n + N - 1) / N, (m + M - 1) / M);
  matrix::multiply_kernel<<<dimGrid, dimBlock>>>(devC, devA, devB, m, p, n);

  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(C, devC, m * n * sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(devA));
  checkCudaErrors(cudaFree(devB));
  checkCudaErrors(cudaFree(devC));
}

} // namespace matrix
