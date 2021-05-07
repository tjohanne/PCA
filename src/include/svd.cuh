#pragma once
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
typedef struct SVD {
    float* U;
    float* S;
    float* V;
} svd_t;
svd_t perform_svd(float* d_A, int m, int n, int economy, const float tolerance, 
        const int max_sweeps, bool verbose, cusolverDnHandle_t cusolverH);
