#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>

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
#define cublasCheckError(ans) ans
#endif
#ifdef DEBUG
#define cusolverCheckError(ans) cusolverAssert((ans), __FILE__, __LINE__);
inline void cusolverAssert(cusolverStatus_t code, const char *file, int line,
                         bool abort = true) {
  if (code != CUSOLVER_STATUS_SUCCESS) {
    fprintf(stderr, "cuSolver Error: %d at %s:%d\n", code, file, line);
    if (abort)
      exit(code);
  }
}
#else
#define cusolverCheckError(ans) ans
#endif