#include "include/svd.cuh"
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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




/**
 * @brief 
 * 
 * @param m 
 * @param n 
 * @param A 
 * @param lda 
 * @param name 
 */
void printMatrix(int m, int n, const float *A, int lda, const char *name) {
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) {
      float Areg = A[row + col * lda];
      printf("%s(%d,%d) = %.3f\n", name, row + 1, col + 1, Areg);
    }
  }
}



/**
 * @brief 
 * 
 * @param m 
 * @param n 
 * @param A 
 * @param lda 
 * @param name 
 */
void printMatrixcsv(int m, int n, const float *A, int lda, const char *name) {
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) {
      float Areg = A[row + col * lda];
      printf("%.3f,", Areg);
    }
    printf("\n");
  }
}

void printVector(int m, const float *A, const char *name) {
  for (int i = 0; i < m; i++) {
    float Areg = A[i];
    printf("%.6f\n", Areg);
    printf("%s(%d) = %.3f\n", name, i, Areg);
  }
}


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
                  const int max_sweeps, bool verbose) {
  cusolverDnHandle_t cusolverH = NULL;
  cudaStream_t stream = NULL;
  gesvdjInfo_t gesvdj_params = NULL;
  cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
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
  status = cusolverDnCreate(&cusolverH);
  assert(CUSOLVER_STATUS_SUCCESS == status);
  cudaCheckError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  status = cusolverDnSetStream(cusolverH, stream);
  assert(CUSOLVER_STATUS_SUCCESS == status);
  status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
  assert(CUSOLVER_STATUS_SUCCESS == status);
  status = cusolverDnXgesvdjSetTolerance(gesvdj_params, tolerance);
  assert(CUSOLVER_STATUS_SUCCESS == status);
  status = cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  cudaCheckError(cudaMalloc((void **)&d_S, sizeof(float) * minmn));
  cudaCheckError(cudaMalloc((void **)&d_U, sizeof(float) * ldu * m));
  cudaCheckError(cudaMalloc((void **)&d_V, sizeof(float) * ldv * n));
  cudaCheckError(cudaMalloc((void **)&d_info, sizeof(int)));
  status = cusolverDnSgesvdj_bufferSize(cusolverH, jobz, economy,
                                        m, //  nrows
                                        n, //  ncols
                                        d_A, lda, d_S, d_U, ldu, d_V, ldv,
                                        &lwork, gesvdj_params);
  assert(CUSOLVER_STATUS_SUCCESS == status);
  cudaCheckError(cudaMalloc((void **)&d_work, sizeof(float) * lwork));

  /* compute SVD */
  status =
      cusolverDnSgesvdj(cusolverH, jobz, economy, m, n, d_A, lda, d_S, d_U, ldu,
                        d_V, ldv, d_work, lwork, d_info, gesvdj_params);
  cudaCheckError(cudaDeviceSynchronize());
  assert(CUSOLVER_STATUS_SUCCESS == status);
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

    status =
        cusolverDnXgesvdjGetSweeps(cusolverH, gesvdj_params, &executed_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);
    status = cusolverDnXgesvdjGetResidual(cusolverH, gesvdj_params, &residual);
    assert(CUSOLVER_STATUS_SUCCESS == status);
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
  if (cusolverH)
    cusolverDnDestroy(cusolverH);
  if (stream)
    cudaStreamDestroy(stream);
  if (gesvdj_params)
    cusolverDnDestroyGesvdjInfo(gesvdj_params);
  SVD svd;
  svd.S = d_S;
  svd.V = d_V;
  svd.U = d_U;
  return svd;
}
