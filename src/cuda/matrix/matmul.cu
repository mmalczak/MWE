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



__global__ static void multiply_kernel_optimized(float *C, float *A, float *B, int32_t m, int32_t p, int32_t n) {
  extern __shared__ float sh_mem[];
  float *As = &sh_mem[0];
  float *Bs = &sh_mem[blockDim.x*blockDim.y];
  int m1 = ((m+blockDim.y-1)/blockDim.y) * blockDim.y;
  int n1 = ((n+blockDim.x-1)/blockDim.x) * blockDim.x;
  int pp = (p+blockDim.x-1)/blockDim.x;

  int idx = threadIdx.x + blockDim.x*threadIdx.y;
  for (int32_t i = blockIdx.y * blockDim.y + threadIdx.y; i < m1; i += gridDim.y * blockDim.y) {
    for (int32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n1; j += gridDim.x * blockDim.x) {

      float s = 0;
      for(int32_t t=0; t<pp;t++) {
        int kx = threadIdx.x + t*blockDim.x;
        int ky = threadIdx.y + t*blockDim.y;
        As[idx] = (i<m && kx<p)?A[i*p+kx]:0;
        Bs[idx] = (j<n && ky<p)?B[j+ky*n]:0;
        __syncthreads();

        for(int k =0; k<blockDim.x; k++) {
          s += As[k+threadIdx.y*blockDim.x] * Bs[k*blockDim.x+threadIdx.x];
        }
        __syncthreads();

      }
      if(i<m && j<n){
        C[i * n + j] = s;
      }
    }
  }
}

void run_multiply_kernel_optimized(float *devC, float *devA, float *devB, int32_t m, int32_t p, int32_t n) {
  int32_t K = 32;
  int32_t gridSizeX = 32;
  int32_t gridSizeY = 32;
  dim3 dimBlock(K, K);
  dim3 dimGrid(gridSizeX, gridSizeY);
  size_t shSize = 2 * K * K* sizeof(float);

  matrix::multiply_kernel_optimized<<<dimGrid, dimBlock, shSize>>>(devC, devA, devB, m, p, n);
}

void multiply_optimized(float *C, float *A, float *B, int32_t m, int32_t p, int32_t n) {
  void *devA, *devB, *devC;
  checkCudaErrors(cudaSetDevice(0));
  checkCudaErrors(cudaMalloc(&devA, m * p * sizeof(float)));
  checkCudaErrors(cudaMalloc(&devB, p * n * sizeof(float)));
  checkCudaErrors(cudaMalloc(&devC, m * n * sizeof(float)));

  checkCudaErrors(cudaMemcpy(devA, A, m * p * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(devB, B, p * n * sizeof(float), cudaMemcpyHostToDevice));

  run_multiply_kernel_optimized(static_cast<float *>(devC), static_cast<float *>(devA), static_cast<float *>(devB), m, p, n);

  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(C, devC, m * n * sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(devA));
  checkCudaErrors(cudaFree(devB));
  checkCudaErrors(cudaFree(devC));
}




} // namespace matrix
