#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "cublas_v2.h"
#include "include/pca.cuh"
#include "include/svd.cuh"
#include <cuda_runtime.h>
#ifndef min
#define min(a, b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a, b) ((a > b) ? a : b)
#endif

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

__global__ void mult_S_U(float *out, float *S, float *U, int features,
                         int samples, int k) {
  // S is a diagonal matrix represented as a vector
  // Note col and row switched since we are dealing with row col order
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < k && row < samples) {
    out[row * k + col] = S[col] * U[col * samples + row];
  }
  __syncthreads();
}

void print_cpu_matrix(int m, int n, const float *A, const char *name) {
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) {
      float Areg = A[col + row * n];
      printf("(%d,%d)%.3f,", row, col, Areg);
    }
    printf("\n");
  }
}

void print_host_matrix(int m, int n, const float *A, const char *name) {
  float *tempmatrix;
  tempmatrix = (float *)malloc(sizeof(float) * m * n);
  cudaMemcpy(tempmatrix, A, sizeof(float) * m * n, cudaMemcpyHostToDevice);
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) {
      float Areg = tempmatrix[col + row * n];
      printf("(%d,%d)%.3f,", row, col, Areg);
    }
    printf("\n");
  }
}

float *mean_shift(float *matrix, int M, int N) {
  cublasHandle_t handle;
  float *x = new float[M];
  float *y = new float[N];
  float *d_matrix = NULL;
  float *clonem = NULL;
  float *d_x = NULL;
  float *d_y = NULL;
  float alpha = 1.0;
  float beta = 0.0;
  for (int i = 0; i < M; i++) {
    x[i] = 1.0f;
  }
  for (int i = 0; i < N; i++) {
    y[i] = 0.0f;
  }
  cudaCheckError(cudaDeviceSynchronize());
  // will need to call cublasDestroy() at some point
  cublasCheckError(cublasCreate(&handle));
  cudaCheckError(cudaDeviceSynchronize());
  cudaCheckError(cudaMalloc((void **)&d_matrix, M * N * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&clonem, M * N * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&d_x, M * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&d_y, N * sizeof(float)));

  cudaCheckError(cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_matrix, matrix, M * N * sizeof(float),
                            cudaMemcpyHostToDevice));


  cudaCheckError(cudaDeviceSynchronize());
  cublasCheckError(cublasSgemv(handle, CUBLAS_OP_N, N, M, &alpha, d_matrix, N,
                               d_x, 1, &beta, d_y, 1));

  cudaCheckError(cudaDeviceSynchronize());
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

  cublasCheckError(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, &alpha,
                               d_matrix, N, &beta, d_matrix, M, clonem, M));
  if (d_y)
    cudaCheckError(cudaFree(d_y));
  if (d_x)
    cudaCheckError(cudaFree(d_x));
  if (d_matrix)
    cudaCheckError(cudaFree(d_matrix));
  if (x)
    free(x);
  if (y)
    free(y);
  return clonem;
}

float *pca_from_S_U(svd_t svd, int M, int N, int k) {
  float *out = NULL;
  float *out_cpu = NULL;
  // Create out matrix
  out_cpu = (float *)malloc(k * M * sizeof(float));
  cudaCheckError(cudaMalloc((void **)&out, k * M * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&out, k * M * sizeof(float)));
  // Create kernel parameters
  int LBLK = 32;
  dim3 tpb(LBLK, LBLK);
  int div = k / LBLK;
  int div2 = M / LBLK;
  if (k % LBLK != 0) {
    div++;
  }
  if (M % LBLK != 0) {
    div2++;
  }
  dim3 bs(div2, div);
  // Call kernel
  mult_S_U<<<bs, tpb>>>(out, svd.S, svd.U, N, M, k);
  cudaCheckError(cudaFree(svd.V));
  cudaCheckError(cudaFree(svd.U));
  cudaCheckError(cudaFree(svd.S));
  cudaCheckError(
      cudaMemcpy(out_cpu, out, k * M * sizeof(float), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaDeviceSynchronize());
  return out_cpu;
}

float_matrix_t perform_pca(float *matrix, int M, int N, int ncomponents, const int econ, const float tol, 
                            const int max_sweeps, const bool verbose, TimeLogger* tl) {
  TimeLogger::timeLog* mean_shift_log;
  TimeLogger::timeLog* perform_svd_log;
  TimeLogger::timeLog* memcpy_log;
  TimeLogger::timeLog* pca_S_U_log;

  if(tl != NULL) 
    mean_shift_log = tl->start("mean_shift()");
  float *d_matrix = mean_shift(matrix, M, N);
  if(tl != NULL) {
    cudaCheckError(cudaDeviceSynchronize());
    tl->stop(mean_shift_log);
    perform_svd_log = tl->start("perform_svd()");
  }
  svd_t svd =
      perform_svd(d_matrix, M, N, econ, tol, max_sweeps, verbose);

  float_matrix_t svd_out;
  if(tl != NULL) {
    cudaCheckError(cudaDeviceSynchronize());
    tl->stop(perform_svd_log);
    memcpy_log = tl->start("svd matrices to device memory");
  }
  int minmn = min(M, N);
  svd_out.S = (float *) malloc(sizeof(float) * minmn);
  svd_out.V = (float *) malloc(sizeof(float) * N * N);
  cudaCheckError(
    cudaMemcpy(svd_out.V, svd.V, N * N * sizeof(float), cudaMemcpyDeviceToHost));
  cudaCheckError(
    cudaMemcpy(svd_out.S, svd.S, minmn * sizeof(float), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaDeviceSynchronize());
  if(tl != NULL) {
    tl->stop(memcpy_log);
    pca_S_U_log = tl->start("pca_from_S_U");
  }
  svd_out.matrix = pca_from_S_U(svd, M, N, ncomponents);
  svd_out.rows = M;
  svd_out.cols = ncomponents;
  if(tl != NULL) {
    cudaCheckError(cudaDeviceSynchronize());
    if(tl != NULL) tl->stop(pca_S_U_log);
  }
  return svd_out;
}