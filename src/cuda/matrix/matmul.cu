#include <cuda.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

namespace matrix {

__global__ static void multiply_kernel(float *C, float *A, float *B, int m, int p, int n) {
  int i = threadIdx.y;
  int j = threadIdx.x;
  int k;
  float s = 0;
  for (k = 0; k < p; k++) {
    s += A[i * p + k] * B[k * n + j];
  }
  C[i * n + j] = s;
}

void multiply(float *C, float *A, float *B, int M, int P, int N) {
  float *devA, *devB, *devC;
  checkCudaErrors(cudaSetDevice(0));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&devA), M * P * sizeof(float)));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&devB), P * N * sizeof(float)));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&devC), M * N * sizeof(float)));

  checkCudaErrors(cudaMemcpy(devA, A, M * P * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devB, B, P * N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 dimBlock(N, M);

  matrix::multiply_kernel<<<1, dimBlock>>>(devC, devA, devB, M, P, N);

  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(C, devC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(devA));
  checkCudaErrors(cudaFree(devB));
  checkCudaErrors(cudaFree(devC));
}

} // namespace matrix
