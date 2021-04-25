#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define M 6
#define N 5
#define IDX2F(i, j, ld) ((((j)) * (ld)) + ((i)))

#include "cublas_v2.h"
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
__global__ void add(int *a, int *b) {
  int i = blockIdx.x;
  if (i < N) {
    b[i] = 2 * a[i];
  }
}

static __inline__ void modify(cublasHandle_t handle, float *m, int ldm, int n,
                              int p, int q, float alpha, float beta) {
  cublasSscal(handle, n - q + 1, &alpha, &m[IDX2F(p, q, ldm)], ldm);
  cublasSscal(handle, ldm - p + 1, &beta, &m[IDX2F(p, q, ldm)], 1);
}

int cublas_example() {
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  int i, j;
  float *devPtrA;
  float *a = 0;
  a = (float *)malloc(M * N * sizeof(*a));
  if (!a) {
    printf("host memory allocation failed");
    return EXIT_FAILURE;
  }
  for (j = 0; j < N; j++) {
    for (i = 0; i < M; i++) {
      a[j * M + i] = (float)((i)*N + j);
    }
  }
  cudaStat = cudaMalloc((void **)&devPtrA, M * N * sizeof(*a));
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation failed");
    return EXIT_FAILURE;
  }
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS initialization failed\n");
    return EXIT_FAILURE;
  }
  stat = cublasSetMatrix(M, N, sizeof(*a), a, M, devPtrA, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download failed");
    cudaFree(devPtrA);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  modify(handle, devPtrA, M, N, 2, 3, 16.0f, 12.0f);
  stat = cublasGetMatrix(M, N, sizeof(*a), devPtrA, M, a, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data upload failed");
    cudaFree(devPtrA);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  cudaFree(devPtrA);
  cublasDestroy(handle);
  for (j = 0; j < N; j++) {
    for (i = 0; i < M; i++) {
      printf("%7.0f", a[j * M + i]);
    }
    printf("\n");
  }
  free(a);
  return EXIT_SUCCESS;
}

void cuda_example() {
  //
  // Create int arrays on the CPU.
  // ('h' stands for "host".)
  //
  int ha[N], hb[N];

  //
  // Create corresponding int arrays on the GPU.
  // ('d' stands for "device".)
  //
  int *da, *db;
  cudaMalloc((void **)&da, N * sizeof(int));
  cudaMalloc((void **)&db, N * sizeof(int));

  //
  // Initialise the input data on the CPU.
  //
  for (int i = 0; i < N; ++i) {
    ha[i] = i;
  }

  //
  // Copy input data to array on GPU.
  //
  cudaMemcpy(da, ha, N * sizeof(int), cudaMemcpyHostToDevice);

  //
  // Launch GPU code with N threads, one per
  // array element.
  //
  add<<<N, 1>>>(da, db);

  //
  // Copy output array from GPU back to CPU.
  //
  cudaMemcpy(hb, db, N * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) {
    printf("%d\n", hb[i]);
  }

  //
  // Free up the arrays on the GPU.
  //
  cudaFree(da);
  cudaFree(db);
}
