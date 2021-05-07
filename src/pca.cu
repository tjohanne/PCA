#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "include/pca.cuh"
#include "include/stats.cuh"
#include "include/svd.cuh"
#include "include/svdapprox.cuh"
#include "include/gpuasserts.cuh"
#ifndef min
#define min(a, b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a, b) ((a > b) ? a : b)
#endif

/**
 * @brief what does this do?
 * 
 * @param out 
 * @param S 
 * @param U 
 * @param features 
 * @param samples 
 * @param k 
 * @return __global__ 
 */
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

/**
 * @brief what does this do?
 * 
 * @param svd 
 * @param M 
 * @param N 
 * @param k 
 * @return float* 
 */
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

/**
 * @brief 
 * 
 * @param matrix 
 * @param M 
 * @param N 
 * @param ncomponents 
 * @param econ 
 * @param tol 
 * @param max_sweeps 
 * @param verbose 
 * @param tl 
 * @return float_matrix_t 
 */
float_matrix_t perform_pca(float *matrix, int M, int N, int ncomponents, const int econ, const float tol, 
                            const int max_sweeps, const bool verbose, TimeLogger* tl) {
  TimeLogger::timeLog* perform_pca_log;
  TimeLogger::timeLog* mean_shift_log;
  TimeLogger::timeLog* perform_svd_log;
  TimeLogger::timeLog* memcpy_log;
  TimeLogger::timeLog* pca_S_U_log;
  // initialize handlers
  cusolverDnHandle_t cusolverH = NULL;
  cublasHandle_t cublasH = NULL;
  cusolverCheckError(cusolverDnCreate(&cusolverH));
  cublasCheckError(cublasCreate(&cublasH));
  if(tl != NULL) {
    perform_pca_log = tl->start("perform_pca()");
    mean_shift_log = tl->start("mean_shift()");
  }
  float *d_matrix = mean_shift(matrix, M, N, cublasH);
  if(tl != NULL) {
    cudaCheckError(cudaDeviceSynchronize());
    tl->stop(mean_shift_log);
    perform_svd_log = tl->start("perform_svd()");
  }
  // svd_t svd =
  //     perform_svd(d_matrix, M, N, econ, tol, max_sweeps, verbose, cusolverH);
  int batch_size = 16;
  svd_t svd =
      perform_svd_approx(d_matrix, M, N, n_components, batch_size, econ, tol, max_sweeps, verbose, cusolverH);

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
    if(tl != NULL) { 
      tl->stop(pca_S_U_log);
      tl->stop(perform_pca_log);
    }
  }
  cublasCheckError(cublasDestroy(cublasH));
  cusolverCheckError(cusolverDnDestroy(cusolverH));
  return svd_out;
}