#include <cuda.h>
#include <cuda_runtime_api.h>
#include "src/cuda/Common/helper_cuda.h"
#include "src/cuda/matrix/include/matmul.hpp"
#include <cstdint>

namespace matrix {

__global__ static void multiply_kernel(float *C, float *A, float *B, int32_t m, int32_t p, int32_t n) {
  extern __shared__ float sh_mem[];
  float *As = &sh_mem[0];
  float *Bs = &sh_mem[blockDim.x * blockDim.y];
  int m1 = ((m + blockDim.y - 1) / blockDim.y) * blockDim.y;
  int n1 = ((n + blockDim.x - 1) / blockDim.x) * blockDim.x;
  int pp = (p + blockDim.x - 1) / blockDim.x;

  int idx = threadIdx.x + blockDim.x * threadIdx.y;
  for (int32_t i = blockIdx.y * blockDim.y + threadIdx.y; i < m1; i += gridDim.y * blockDim.y) {
    for (int32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n1; j += gridDim.x * blockDim.x) {

      float s = 0;
      for (int32_t t = 0; t < pp; t++) {
        int kx = threadIdx.x + t * blockDim.x;
        int ky = threadIdx.y + t * blockDim.y;
        As[idx] = (i < m && kx < p) ? A[i * p + kx] : 0;
        Bs[idx] = (j < n && ky < p) ? B[j + ky * n] : 0;
        __syncthreads();

        for (int k = 0; k < blockDim.x; k++) {
          s += As[k + threadIdx.y * blockDim.x] * Bs[k * blockDim.x + threadIdx.x];
        }
        __syncthreads();
      }
      if (i < m && j < n) {
        C[i * n + j] = s;
      }
    }
  }
}

void run_multiply_kernel(float *devC, float *devA, float *devB, int32_t m, int32_t p, int32_t n) {
  int32_t K = 32;
  int32_t gridSizeX = 32;
  int32_t gridSizeY = 32;
  dim3 dimBlock(K, K);
  dim3 dimGrid(gridSizeX, gridSizeY);
  size_t shSize = 2 * K * K * sizeof(float);

  matrix::multiply_kernel<<<dimGrid, dimBlock, shSize>>>(devC, devA, devB, m, p, n);
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

///// Forward declaration of the matrix multiplication kernel
///__global__ void MatMulKernel(const Matrix2, const Matrix2, Matrix2);
///
///// Matrix multiplication - Host code
///// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
/// void MatMul(const Matrix2 A, const Matrix2 B, Matrix2 C) {
///  // Load A and B to device memory
///  Matrix2 d_A;
///  d_A.width = A.width;
///  d_A.height = A.height;
///  size_t size = A.width * A.height * sizeof(float);
///  cudaMalloc(&d_A.elements, size);
///  cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
///  Matrix2 d_B;
///  d_B.width = B.width;
///  d_B.height = B.height;
///  size = B.width * B.height * sizeof(float);
///  cudaMalloc(&d_B.elements, size);
///  cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
///
///  // Allocate C in device memory
///  Matrix2 d_C;
///  d_C.width = C.width;
///  d_C.height = C.height;
///  size = C.width * C.height * sizeof(float);
///  cudaMalloc(&d_C.elements, size);
///
///  // Invoke kernel
///  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
///  dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
///  MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
///
///  // Read C from device memory
///  cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
///
///  // Free device memory
///  cudaFree(d_A.elements);
///  cudaFree(d_B.elements);
///  cudaFree(d_C.elements);
///}
///
///// Matrix multiplication kernel called by MatMul()
///__global__ void MatMulKernel(Matrix2 A, Matrix2 B, Matrix2 C) {
///  // Each thread computes one element of C
///  // by accumulating results into Cvalue
///  float Cvalue = 0;
///  int row = blockIdx.y * blockDim.y + threadIdx.y;
///  int col = blockIdx.x * blockDim.x + threadIdx.x;
///  for (int e = 0; e < A.width; ++e)
///    Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
///  C.elements[row * C.width + col] = Cvalue;
///}
///


__device__ float GetElement(const Matrix2 A, int row, int col)
{
      return A.elements[row * A.stride + col];
}
// Set a matrix element
__device__ void SetElement(Matrix2 A, int row, int col,
                               float value)
{
      A.elements[row * A.stride + col] = value;
}

__device__ Matrix2 GetSubMatrix(Matrix2 A, int row, int col) {
  Matrix2 Asub;
  Asub.width = BLOCK_SIZE;
  Asub.height = BLOCK_SIZE;
  Asub.stride = A.stride;
  Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
  return Asub;
}
// Thread block size
#define BLOCK_SIZE 16
// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix2, const Matrix2, Matrix2);
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix2 A, const Matrix2 B, Matrix2 C) {
  // Load A and B to device memory
  Matrix2 d_A;
  d_A.width = d_A.stride = A.width;
  d_A.height = A.height;
  size_t size = A.width * A.height * sizeof(float);
  cudaMalloc(&d_A.elements, size);
  cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
  Matrix2 d_B;
  d_B.width = d_B.stride = B.width;
  d_B.height = B.height;
  size = B.width * B.height * sizeof(float);
  cudaMalloc(&d_B.elements, size);
  cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
  // Allocate C in device memory
  Matrix2 d_C;
  d_C.width = d_C.stride = C.width;
  d_C.height = C.height;
  size = C.width * C.height * sizeof(float);
  cudaMalloc(&d_C.elements, size);
  // Invoke kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
  MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
  // Read C from device memory
  cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(d_A.elements);
  cudaFree(d_B.elements);
  cudaFree(d_C.elements);
}
// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix2 A, Matrix2 B, Matrix2 C) {
  // Block row and column
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  // Each thread block computes one sub-matrix Csub of C
  Matrix2 Csub = GetSubMatrix(C, blockRow, blockCol);
  // Each thread computes one element of Csub
  // by accumulating results into Cvalue
  float Cvalue = 0;
  // Thread row and column within Csub
  int row = threadIdx.y;
  int col = threadIdx.x;
  // Loop over all the sub-matrices of A and B that are
  // required to compute Csub
  // Multiply each pair of sub-matrices together
  // and accumulate the results
  for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
    // Get sub-matrix Asub of A
    Matrix2 Asub = GetSubMatrix(A, blockRow, m);
    // Get sub-matrix Bsub of B
    Matrix2 Bsub = GetSubMatrix(B, m, blockCol);
    // Shared memory used to store Asub and Bsub respectively
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    // Load Asub and Bsub from device memory to shared memory
    // Each thread loads one element of each sub-matrix
    As[row][col] = GetElement(Asub, row, col);
    Bs[row][col] = GetElement(Bsub, row, col);
    // Synchronize to make sure the sub-matrices are loaded
    // before starting the computation
    __syncthreads();
    // Multiply Asub and Bsub together
    for (int e = 0; e < BLOCK_SIZE; ++e)
      Cvalue += As[row][e] * Bs[e][col];
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }
  // Write Csub to device memory
  // Each thread writes one element
  SetElement(Csub, row, col, Cvalue);
}

} // namespace matrix
