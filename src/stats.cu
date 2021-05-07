#include "include/stats.cuh"
#include "include/gpuasserts.cuh"

/**
 * @brief Takes a matrix thats been summed r
 * 
 * @param total 
 * @param n 
 * @param m 
 * @return __global__ 
 */
__global__ void get_average_from_total(float *total, int n, int m) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n) {
    total[row] = total[row] / m;
  }
  __syncthreads();
}

/**
 * @brief subtracts average vector from each matrix entry 
 * 
 * @param matrix matrix of size m x n
 * @param averages 
 * @param m 
 * @param n 
 */
__global__ void subtract(float *matrix, float *averages, int m, int n) {
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < n && row < m) {
    matrix[row * n + col] = matrix[row * n + col] - averages[col];
  }
  __syncthreads();
}


/**
 * @brief Centers the original input matrix by computing 
 * the mean for each feature, and subtracting the mean 
 * from each observation.
 * 
 * @param matrix 
 * @param M 
 * @param N 
 * @return float* 
 */
float *mean_shift(float *matrix, int M, int N, cublasHandle_t handle) {
  float *x = new float[M];
  float *y = new float[N];
  float *d_matrix = NULL;
  float *clonem = NULL;
  float *d_x = NULL;
  float *d_y = NULL;
  float alpha = 1.0;
  float beta = 0.0;
  for (int i = 0; i < M; i++) {
    x[i] = 1.0f;
  }
  for (int i = 0; i < N; i++) {
    y[i] = 0.0f;
  }
  cudaCheckError(cudaMalloc((void **)&d_matrix, M * N * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&clonem, M * N * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&d_x, M * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&d_y, N * sizeof(float)));

  cudaCheckError(cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_matrix, matrix, M * N * sizeof(float),
                            cudaMemcpyHostToDevice));


  cudaCheckError(cudaDeviceSynchronize());
  cublasCheckError(cublasSgemv(handle, CUBLAS_OP_N, N, M, &alpha, d_matrix, N,
                               d_x, 1, &beta, d_y, 1));

  cudaCheckError(cudaDeviceSynchronize());
  const int threadsPerBlock = 512;
  int blocks = N / threadsPerBlock;
  if (N % threadsPerBlock != 0) {
    blocks++;
  }
  int LBLK = 32;
  dim3 tpb(LBLK, LBLK);
  int div = N / LBLK;
  int div2 = M / LBLK;
  if (N % LBLK != 0) {
    div++;
  }
  if (M % LBLK != 0) {
    div2++;
  }
  dim3 bs(div2, div);
  get_average_from_total<<<blocks, threadsPerBlock>>>(d_y, N, M);
  cudaCheckError(cudaDeviceSynchronize());
  subtract<<<bs, tpb>>>(d_matrix, d_y, M, N);
  cudaCheckError(cudaDeviceSynchronize());

  cublasCheckError(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, &alpha,
                               d_matrix, N, &beta, d_matrix, M, clonem, M));
  if (d_y)
    cudaCheckError(cudaFree(d_y));
  if (d_x)
    cudaCheckError(cudaFree(d_x));
  if (d_matrix)
    cudaCheckError(cudaFree(d_matrix));
  if (x)
    free(x);
  if (y)
    free(y);
  return clonem;
}