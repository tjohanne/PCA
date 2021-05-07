
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
__global__ void get_average_from_total(float *total, int n, int m);
float *mean_shift(float *matrix, int M, int N, cublasHandle_t handle);