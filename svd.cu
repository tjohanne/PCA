/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include gesvdj_example.cpp
 *   g++ -o gesvdj_example gesvdj_example.o -L/usr/local/cuda/lib64 -lcudart
 * -lcusolver
 *   TODO check nvcc flag?
 */
#include "svd.cuh"
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
void printMatrix(int m, int n, const float *A, int lda, const char *name) {
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) {
      float Areg = A[row + col * lda];
      printf("%s(%d,%d) = %.3f\n", name, row + 1, col + 1, Areg);
    }
  }
}

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

__global__ void vec_to_diag(float *vec, float *diag_mat, int vec_length) {
  int diag_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (diag_index < vec_length) {
    diag_mat[vec_length * diag_index + diag_index] = vec[diag_index];
  }
  __syncthreads();
}

svd_t perform_svd(float *d_A, int m, int n) {
  cusolverDnHandle_t cusolverH = NULL;
  cudaStream_t stream = NULL;
  gesvdjInfo_t gesvdj_params = NULL;

  cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
  cudaError_t cudaStat1 = cudaSuccess;
  cudaError_t cudaStat2 = cudaSuccess;
  cudaError_t cudaStat3 = cudaSuccess;
  cudaError_t cudaStat4 = cudaSuccess;
  cudaError_t cudaStat5 = cudaSuccess;
  const int lda = m; /* A is m-by-n */
  const int ldu = m; /* U is m-by-m */
  const int ldv = n; /* V is n-by-n */
  const int minmn = min(m, n);
  /*       | 1 2  |
   *   A = | 4 5  |
   *       | 2 1  |
   */
  //   float A[lda * n] = {4.0, 0.0, 3.0, -5.0};
  float *U = new float[ldu * m];
  float *V = new float[ldv * n];
  float *S = new float[minmn * minmn];
  // float U[ldu*m]; /* m-by-m unitary matrix, left singular vectors  */
  // float V[ldv*n]; /* n-by-n unitary matrix, right singular vectors */
  // float S[minmn];     /* numerical singular value */
  /* exact singular values */
  //  TODO s_exact is for testing, remove
  float S_exact[2 * 3] = {6.3, 3.16};
  //   float *d_A = NULL;    /* device copy of A */
  float *d_S = NULL; /* singular values */
  float *d_U = NULL; /* left singular vectors */
  float *d_V = NULL; /* right singular vectors */
  float *d_Smat = NULL;
  int *d_info = NULL;   /* error info */
  int lwork = 0;        /* size of workspace */
  float *d_work = NULL; /* devie workspace for gesvdj */
  int info = 0;         /* host copy of error info */
                        /* configuration of gesvdj  */
  const float tol = 1.e-7;
  const int max_sweeps = 15;
  const cusolverEigMode_t jobz =
      CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
  const int econ = 0;           /* econ = 1 for economy size */

  /* numerical results of gesvdj  */
  double residual = 0;
  int executed_sweeps = 0;

  printf("example of gesvdj \n");
  printf("tol = %E, default value is machine zero \n", tol);
  printf("max. sweeps = %d, default value is 100\n", max_sweeps);
  printf("econ = %d \n", econ);

  printf("A = (matlab base-1)\n");
  //   printMatrix(m, n, A, lda, "A");
  printf("=====\n");

  /* step 1: create cusolver handle, bind a stream */
  status = cusolverDnCreate(&cusolverH);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  assert(cudaSuccess == cudaStat1);

  status = cusolverDnSetStream(cusolverH, stream);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  /* step 2: configuration of gesvdj */
  status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  /* default value of tolerance is machine zero */
  status = cusolverDnXgesvdjSetTolerance(gesvdj_params, tol);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  /* default value of max. sweeps is 100 */
  status = cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  /* step 3: copy A and B to device */
  // cudaStat1 = cudaMalloc((void **)&d_A, sizeof(float) * lda * n);
  cudaStat2 = cudaMalloc((void **)&d_S, sizeof(float) * minmn);
  cudaStat3 = cudaMalloc((void **)&d_U, sizeof(float) * ldu * m);
  cudaStat4 = cudaMalloc((void **)&d_V, sizeof(float) * ldv * n);
  cudaStat5 = cudaMalloc((void **)&d_info, sizeof(int));
  cudaStat5 = cudaMalloc((void **)&d_Smat, sizeof(float) * minmn * minmn);
  assert(cudaSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat2);
  assert(cudaSuccess == cudaStat3);
  assert(cudaSuccess == cudaStat4);
  assert(cudaSuccess == cudaStat5);
  assert(cudaSuccess == cudaStat5);

  //   cudaStat1 =
  //   cudaMemcpy(d_A, A, sizeof(float) * lda * n, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStat1);
  /* step 4: query workspace of SVD */
  status = cusolverDnSgesvdj_bufferSize(
      cusolverH,
      jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
      /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors
       */
      econ, /* econ = 1 for economy size */
      m,    /* nubmer of rows of A, 0 <= m */
      n,    /* number of columns of A, 0 <= n  */
      d_A,  /* m-by-n */
      lda,  /* leading dimension of A */
      d_S,  /* min(m,n) */
            /* the singular values in descending order */
      d_U,  /* m-by-m if econ = 0 */
            /* m-by-min(m,n) if econ = 1 */
      ldu,  /* leading dimension of U, ldu >= max(1,m) */
      d_V,  /* n-by-n if econ = 0  */
            /* n-by-min(m,n) if econ = 1  */
      ldv,  /* leading dimension of V, ldv >= max(1,n) */
      &lwork, gesvdj_params);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  cudaStat1 = cudaMalloc((void **)&d_work, sizeof(float) * lwork);
  assert(cudaSuccess == cudaStat1);

  /* step 5: compute SVD */
  status = cusolverDnSgesvdj(
      cusolverH,
      jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
      /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors
       */
      econ, /* econ = 1 for economy size */
      m,    /* nubmer of rows of A, 0 <= m */
      n,    /* number of columns of A, 0 <= n  */
      d_A,  /* m-by-n */
      lda,  /* leading dimension of A */
      d_S,  /* min(m,n)  */
            /* the singular values in descending order */
      d_U,  /* m-by-m if econ = 0 */
            /* m-by-min(m,n) if econ = 1 */
      ldu,  /* leading dimension of U, ldu >= max(1,m) */
      d_V,  /* n-by-n if econ = 0  */
            /* n-by-min(m,n) if econ = 1  */
      ldv,  /* leading dimension of V, ldv >= max(1,n) */
      d_work, lwork, d_info, gesvdj_params);
  cudaStat1 = cudaDeviceSynchronize();
  assert(CUSOLVER_STATUS_SUCCESS == status);
  assert(cudaSuccess == cudaStat1);

  const int threadsPerBlock = 64;
  int blocks = minmn / threadsPerBlock;
  if (minmn % threadsPerBlock != 0) {
    blocks++;
  }

  //  transform S from a vector to a diagonal matrix
  vec_to_diag<<<1, threadsPerBlock>>>(d_S, d_Smat, minmn);

  cudaStat1 =
      cudaMemcpy(U, d_U, sizeof(float) * ldu * m, cudaMemcpyDeviceToHost);
  cudaStat2 =
      cudaMemcpy(V, d_V, sizeof(float) * ldv * n, cudaMemcpyDeviceToHost);
  cudaStat3 = cudaMemcpy(S, d_Smat, sizeof(float) * minmn * minmn,
                         cudaMemcpyDeviceToHost);
  cudaStat4 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
  cudaStat5 = cudaDeviceSynchronize();
  assert(cudaSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat2);
  assert(cudaSuccess == cudaStat3);
  assert(cudaSuccess == cudaStat4);
  assert(cudaSuccess == cudaStat5);

  if (0 == info) {
    printf("gesvdj converges \n");
  } else if (0 > info) {
    printf("%d-th parameter is wrong \n", -info);
    exit(1);
  } else {
    printf("WARNING: info = %d : gesvdj does not converge \n", info);
  }

  printf("S = singular values (matlab base-1)\n");
  printMatrix(minmn, 1, S, minmn, "S");
  printf("=====\n");

  printf("U = left singular vectors (matlab base-1)\n");
  printMatrix(m, m, U, ldu, "U");
  printf("=====\n");

  printf("V = right singular vectors (matlab base-1)\n");
  printMatrix(n, n, V, ldv, "V");
  printf("=====\n");

  printf("S = matrix (matlab base-1)\n");
  printMatrix(minmn, minmn, S, minmn, "S MATRIX");
  printf("=====\n");

  /* step 6: measure error of singular value */
  float ds_sup = 0;
  for (int j = 0; j < minmn; j++) {
    float err = fabs(S[j] - S_exact[j]);
    ds_sup = (ds_sup > err) ? ds_sup : err;
  }
  printf("|S - S_exact|_sup = %E \n", ds_sup);

  status =
      cusolverDnXgesvdjGetSweeps(cusolverH, gesvdj_params, &executed_sweeps);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  status = cusolverDnXgesvdjGetResidual(cusolverH, gesvdj_params, &residual);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  printf("residual |A - U*S*V**H|_F = %E \n", residual);
  printf("number of executed sweeps = %d \n", executed_sweeps);

  /*  free resources  */
  if (d_A)
    cudaFree(d_A);
  if (d_S)
    cudaFree(d_S);
  //   if (d_U)
  //     cudaFree(d_U);
  //   if (d_V)
  //     cudaFree(d_V);
  if (d_info)
    cudaFree(d_info);
  if (d_work)
    cudaFree(d_work);
  if (U)
    free(U);
  if (V)
    free(V);
  if (S)
    free(S);
  if (cusolverH)
    cusolverDnDestroy(cusolverH);
  if (stream)
    cudaStreamDestroy(stream);
  if (gesvdj_params)
    cusolverDnDestroyGesvdjInfo(gesvdj_params);
  SVD svd;
  svd.S = d_Smat;
  svd.V = d_V;
  svd.U = d_U;
  // cudaDeviceReset();
  return svd;
}
