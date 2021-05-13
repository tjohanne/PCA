
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
__global__ void get_average_from_total(float *total, int n, int m);
float *mean_shift(float *matrix, int M, int N, int batch_size, cublasHandle_t handle);
float* row_to_column_order(float *d_matrix, int M, int N, int batch_size, cublasHandle_t handle);