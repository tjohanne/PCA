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
  for (int i = 0; i < M; i++) {
    x[i] = 1.0f;
  }
  for (int i = 0; i < N; i++) {
    y[i] = 0.0f;
  }
  printVector(M, x, "X");
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
  cudaMemcpy(x, d_x, sizeof(float) * M, cudaMemcpyDeviceToHost);
  cudaMemcpy(y, d_y, sizeof(float) * N, cudaMemcpyDeviceToHost);
  printVector(M, x, "X");
  printVector(N, y, "Y");
  // printMatrix(M, N, )
  return d_matrix;
}

void perform_pca(float *matrix, int M, int N) {
  mean_shift(matrix, M, N);
  // matrix[0] = 4.0;
  // matrix[1] = 0.0;
  // matrix[2] = 3.0;
  // matrix[3] = -5.0;
  // svd_t svd = perform_svd(matrix, 2, 2);
}