#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cublas_v2.h"
#include "svd.cuh"
#include <cuda_runtime.h>
#ifndef min
#define min(a, b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a, b) ((a > b) ? a : b)
#endif
//
// Nearly minimal CUDA example.
// Compile with:
//
// nvcc -o example example.cu
//

//
// A function marked __global__
// runs on the GPU but can be called from
// the CPU.
//
// This function multiplies the elements of an array
// of ints by 2.
//
// The entire computation can be thought of as running
// with one thread per array element with blockIdx.x
// identifying the thread.
//
// The comparison i<N is because often it isn't convenient
// to have an exact 1-1 correspondence between threads
// and array elements. Not strictly necessary here.
//
// Note how we're mixing GPU and CPU code in the same source
// file. An alternative way to use CUDA is to keep
// C/C++ code separate from CUDA code and dynamically
// compile and load the CUDA code at runtime, a little
// like how you compile and load OpenGL shaders from
// C/C++ code.
//
#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
  printf("Cuda assert no error\n");
}
#else
#define cudaCheckError(ans) ans
#endif
#define DEBUG
#ifdef DEBUG
#define cublasCheckError(ans) cublasAssert((ans), __FILE__, __LINE__);
inline void cublasAssert(cublasStatus_t code, const char *file, int line,
                         bool abort = true) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLAS Error: %d at %s:%d\n", code, file, line);
    if (abort)
      exit(code);
  }
  printf("Cuda assert no error\n");
}
#else
#define cudaCheckError(ans) ans
#endif

__global__ void get_average_from_total(float *total, int n, int m) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n) {
    total[row] = total[row] / m;
  }
  __syncthreads();
}

__global__ void subtract(float *matrix, float *averages, int m, int n) {
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < n && row < m) {
    matrix[row * n + col] = matrix[row * n + col] - averages[col];
  }
  __syncthreads();
}

void print_cpu_matrix(int m, int n, const float *A, const char *name) {
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) {
      float Areg = A[col + row * n];
      printf("%.3f,", Areg);
    }
    printf("\n");
  }
}

float *mean_shift(float *matrix, int M, int N) {
  cublasHandle_t handle;
  float *x = new float[M];
  float *y = new float[N];
  float *d_matrix = NULL;
  float *d_x = NULL;
  float *d_y = NULL;
  float alpha = 1.0;
  float beta = 0.0;
  // float *alpha = new float[1];
  // *alpha = 1.0f;
  // float *beta = new float[1];
  // *beta = 1.0f;
  printMatrix(M, N, matrix, M, "matrix");
  for (int i = 0; i < M; i++) {
    x[i] = 1.0f;
  }
  for (int i = 0; i < N; i++) {
    y[i] = 0.0f;
  }
  // will need to call cublasDestroy() at some point
  cublasCheckError(cublasCreate(&handle));
  cudaCheckError(cudaMalloc((void **)&d_matrix, M * N * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&d_x, M * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&d_y, N * sizeof(float)));
  cudaCheckError(cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_matrix, matrix, M * N * sizeof(float),
                            cudaMemcpyHostToDevice));
  // or CUBLAS_OP_T?
  cublasCheckError(cublasSgemv(handle, CUBLAS_OP_N, N, M, &alpha, d_matrix, N,
                               d_x, 1, &beta, d_y, 1));
  const int threadsPerBlock = 512;
  int blocks = N / threadsPerBlock;
  if (N % threadsPerBlock != 0) {
    blocks++;
  }
  int LBLK = 32;
  dim3 tpb(LBLK, LBLK);
  int div = N / LBLK;
  int div2 = M / LBLK;
  if (N % LBLK != 0) {
    div++;
  }
  if (M % LBLK != 0) {
    div2++;
  }
  dim3 bs(div2, div);
  get_average_from_total<<<blocks, threadsPerBlock>>>(d_y, N, M);
  cudaCheckError(cudaDeviceSynchronize());
  subtract<<<bs, tpb>>>(d_matrix, d_y, M, N);
  cudaCheckError(cudaDeviceSynchronize());
  if (d_y)
    cudaCheckError(cudaFree(d_y));
  if (d_x)
    cudaCheckError(cudaFree(d_x));
  if (x)
    free(x);
  if (y)
    free(y);
  // print_cpu_matrix(M, N, matrix, "matrix");
  return d_matrix;
}

void perform_pca(float *matrix, int M, int N) {
  float *d_matrix = mean_shift(matrix, M, N);
  svd_t svd = perform_svd(d_matrix, M, N);
}