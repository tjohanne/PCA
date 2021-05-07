#include "include/svd.cuh"
#include "include/stats.cuh"
#include "include/gpuasserts.cuh"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef min
#define min(a, b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a, b) ((a > b) ? a : b)
#endif

/**
 * @brief 
 * 
 * @param d_A 
 * @param m 
 * @param n 
 * @param economy 
 * @param tolerance 
 * @param max_sweeps 
 * @param verbose 
 * @return svd_t 
 */

svd_t perform_svd(float *d_A, int m, int n, int economy, const float tolerance,
                  const int max_sweeps, bool verbose, cusolverDnHandle_t cusolverH) {
  cudaStream_t stream = NULL;
  gesvdjInfo_t gesvdj_params = NULL;
  const int lda = m;
  const int ldu = m;
  const int ldv = n;
  const int minmn = min(m, n);
  float *d_S = NULL;
  float *d_U = NULL;
  float *d_V = NULL;
  int *d_info = NULL;   /* error info */
  int lwork = 0;        /* size of workspace */
  float *d_work = NULL; /* devie workspace for gesvdj */
  int info = 0;         /* host copy of error info */
  const cusolverEigMode_t jobz =
      CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
  double residual = 0;
  int executed_sweeps = 0;
  /* create cusolver handle */
  
  cudaCheckError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cusolverCheckError(cusolverDnSetStream(cusolverH, stream));
  cusolverCheckError(cusolverDnCreateGesvdjInfo(&gesvdj_params));
  cusolverCheckError(cusolverDnXgesvdjSetTolerance(gesvdj_params, tolerance));
  cusolverCheckError(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

  cudaCheckError(cudaMalloc((void **)&d_S, sizeof(float) * minmn));
  cudaCheckError(cudaMalloc((void **)&d_U, sizeof(float) * ldu * m));
  cudaCheckError(cudaMalloc((void **)&d_V, sizeof(float) * ldv * n));
  cudaCheckError(cudaMalloc((void **)&d_info, sizeof(int)));
  cusolverCheckError(cusolverDnSgesvdj_bufferSize(cusolverH, jobz, economy,
                                        m, //  nrows
                                        n, //  ncols
                                        d_A, lda, d_S, d_U, ldu, d_V, ldv,
                                        &lwork, gesvdj_params));
  cudaCheckError(cudaMalloc((void **)&d_work, sizeof(float) * lwork));

  /* compute SVD */
  cusolverCheckError(
      cusolverDnSgesvdj(cusolverH, jobz, economy, m, n, d_A, lda, d_S, d_U, ldu,
                        d_V, ldv, d_work, lwork, d_info, gesvdj_params));
  cudaCheckError(cudaDeviceSynchronize());
  cudaCheckError(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaDeviceSynchronize());

  if (0 == info) {
    printf("gesvdj converges \n");
  } else if (0 > info) {
    printf("%d-th parameter is wrong \n", -info);
    exit(1);
  } else {
    printf("WARNING: info = %d : gesvdj does not converge \n", info);
  }

  if (verbose) {

    cusolverCheckError(
        cusolverDnXgesvdjGetSweeps(cusolverH, gesvdj_params, &executed_sweeps));
    cusolverCheckError(cusolverDnXgesvdjGetResidual(cusolverH, gesvdj_params, &residual));
    printf("residual |A - U*S*V**H|_F = %E \n", residual);
    printf("number of executed sweeps = %d \n", executed_sweeps);
  }
  /*  free resources  */
  if (d_A)
    cudaFree(d_A);
  if (d_info)
    cudaFree(d_info);
  if (d_work)
    cudaFree(d_work);
  if (stream)
    cudaStreamDestroy(stream);
  if (gesvdj_params)
    cusolverDnDestroyGesvdjInfo(gesvdj_params);
  svd_t svd;
  svd.S = d_S;
  svd.V = d_V;
  svd.U = d_U;
  return svd;
}
