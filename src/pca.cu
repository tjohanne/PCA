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
#include "debugFunctions.cu"
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
float_matrix_t perform_pca(float *matrix, int M, int N, int n_components, const int econ, const float tol, 
                            const int max_sweeps, const bool verbose, TimeLogger* tl, std::string solver) {
  TimeLogger::timeLog* perform_pca_log;
  TimeLogger::timeLog* mean_shift_log;
  TimeLogger::timeLog* perform_svd_log;
  TimeLogger::timeLog* memcpy_log;
  TimeLogger::timeLog* pca_S_U_log;
  svd_t svd;
  // initialize handlers
  cusolverDnHandle_t cusolverH = NULL;
  cublasHandle_t cublasH = NULL;
  cusolverCheckError(cusolverDnCreate(&cusolverH));
  cublasCheckError(cublasCreate(&cublasH));
  int batch_size = 1;
  if(solver == "approx") {
    batch_size = 1;
  }
  if(tl != NULL) {
    perform_pca_log = tl->start("perform_pca()");
    mean_shift_log = tl->start("mean_shift()");
  }
  float *d_matrix = mean_shift(matrix, M, N, 1, cublasH);


    float A[12] = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 10.0, 8.0, 6.0, 9.0, 7.0, 5.0};
    float *a = new float[12];
    int mrows = 6;
    int ncols = 2;
    memcpy(a, A, 12 * sizeof(float));
    print_cpu_matrix(mrows, ncols, A);
    float *d_A = NULL;
    cudaCheckError(cudaMalloc((void **)&d_A, 12 * sizeof(float)));
    cudaCheckError(cudaMemcpy(d_A, a, 12 * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaDeviceSynchronize());


  if(tl != NULL) {
    cudaCheckError(cudaDeviceSynchronize());
    tl->stop(mean_shift_log);
    perform_svd_log = tl->start("perform_svd()");
  }
  printf("Solver %s\n", solver.c_str());
  if(solver == "jacobi") {
    svd = perform_svd(d_matrix, M, N, econ, tol, max_sweeps, verbose, cusolverH);
  }
  else if(solver == "approx") {
    // float *d_A, int m, int n, int batch_size, int economy, const float tolerance,
                  // const int max_sweeps, bool verbose, cusolverDnHandle_t cusolverH);
    svd =
        perform_svd_approx(d_matrix, M, N, n_components, batch_size, econ, tol, max_sweeps, verbose, cusolverH);
  }
  else {
    std::cout << "Wrong solver specified " << std::endl;
    exit(1);
  }
  print_device_vector(4, d_A);
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
  svd_out.matrix = pca_from_S_U(svd, M, N, n_components);
  svd_out.rows = M;
  svd_out.cols = n_components;
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