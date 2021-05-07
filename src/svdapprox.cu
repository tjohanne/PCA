#include "include/svdapprox.cuh"
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
svd_t perform_svd_approx(float *d_A, int m, int n, int n_components, int batch_size, int economy, const float tolerance,
                  const int max_sweeps, bool verbose, cusolverDnHandle_t cusolverH) {
    cudaStream_t stream = NULL;
    m /= batch_size; // TODO is this needed?
    const int lda = m;
    const int ldu = m;
    const int ldv = n;
    const int rank = n;
    const long long int strideA = (long long int)lda*n;
    const long long int strideS = n;
    const long long int strideU = (long long int)ldu*n;
    const long long int strideV = (long long int)ldv*n;
    /* | 1 2 | | 10 9 |
    * A0 = | 4 5 |, A1 = | 8 7 |
    * | 2 1 | | 6 5 |
    */
    float *d_S = NULL; /* singular values */
    float *d_U = NULL; /* left singular vectors */
    float *d_V = NULL; /* right singular vectors */
    int *d_info = NULL; /* error info */
    int lwork = 0; /* size of workspace */
    float *d_work = NULL; /* devie workspace for gesvda */
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
    double* RnrmF = new double[batch_size]; /* residual norm */
    int* info = new int[batch_size]; /* host copy of error info */
    /* step 1: create cusolver handle, bind a stream */
    cusolverCheckError(cusolverDnCreate(&cusolverH));
    cudaCheckError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cusolverCheckError(cusolverDnSetStream(cusolverH, stream));
    /* step 2: copy A to device */
    cudaCheckError(cudaMalloc ((void**)&d_S , sizeof(float)*strideS*batch_size));
    cudaCheckError(cudaMalloc ((void**)&d_U , sizeof(float)*strideU*batch_size));
    cudaCheckError(cudaMalloc ((void**)&d_V , sizeof(float)*strideV*batch_size));
    cudaCheckError(cudaMalloc ((void**)&d_info, sizeof(int)*batch_size));
    cudaCheckError(cudaDeviceSynchronize()); /* sync with null stream */
    /* step 3: query workspace of SVD */
    cusolverCheckError(cusolverDnSgesvdaStridedBatched_bufferSize(
    cusolverH,
    jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
    /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular
    vectors */
    rank, /* number of singular values */
    m, /* nubmer of rows of Aj, 0 <= m */
    n, /* number of columns of Aj, 0 <= n */
    d_A, /* Aj is m-by-n */
    lda, /* leading dimension of Aj */
    strideA, /* >= lda*n */
    d_S, /* Sj is rank-by-1, singular values in descending order */
    strideS, /* >= rank */
    d_U, /* Uj is m-by-rank */
    ldu, /* leading dimension of Uj, ldu >= max(1,m) */
    strideU, /* >= ldu*rank */
    d_V, /* Vj is n-by-rank */
    ldv, /* leading dimension of Vj, ldv >= max(1,n) */
    strideV, /* >= ldv*rank */
    &lwork,
    batch_size /* number of matrices */
    ));
    cudaCheckError(cudaMalloc((void**)&d_work , sizeof(float)*lwork));
    /* step 4: compute SVD */
    cusolverCheckError(cusolverDnSgesvdaStridedBatched(
        cusolverH,
        jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
        /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular
        vectors */
        rank, /* number of singular values */
        m, /* nubmer of rows of Aj, 0 <= m */
        n, /* number of columns of Aj, 0 <= n */
        d_A, /* Aj is m-by-n */
        lda, /* leading dimension of Aj */
        strideA, /* >= lda*n */
        d_S, /* Sj is rank-by-1 */
        /* the singular values in descending order */
        strideS, /* >= rank */
        d_U, /* Uj is m-by-rank */
        ldu, /* leading dimension of Uj, ldu >= max(1,m) */
        strideU, /* >= ldu*rank */
        d_V, /* Vj is n-by-rank */
        ldv, /* leading dimension of Vj, ldv >= max(1,n) */
        strideV, /* >= ldv*rank */
        d_work,
        lwork,
        d_info,
        RnrmF,
        batch_size /* number of matrices */
    ));
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaMemcpy(info, d_info, sizeof(int)*batch_size,
    cudaMemcpyDeviceToHost));
    cudaCheckError(cudaDeviceSynchronize()); //TODO remove info?
    if ( 0 > info[0] ){
        printf("%d-th parameter is wrong \n", -info[0]);
        exit(1);
    }
    /* free resources */
    if (d_A ) cudaFree(d_A);
    if (d_info ) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);
    if (stream ) cudaStreamDestroy(stream);
    SVD svd;
    // print_host_matrix()
    svd.S = d_S;
    svd.V = d_V;
    svd.U = d_U;
    return svd;
}