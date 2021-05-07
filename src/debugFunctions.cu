#include <stdio.h>
#include <cuda_runtime.h>

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
 * @brief Prints m x n matrix A on host memory. 
 */
void print_cpu_matrix(int m, int n, const float *A) {
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) {
      float Areg = A[col + row * n];
      printf("(%d,%d)%.3f,", row, col, Areg);
    }
    printf("\n");
  }
}

/**
 * @brief Prints m x n matrix A on device memory. 
 */
void print_device_matrix(int m, int n, const float *A) {
  float *tempmatrix;
  tempmatrix = (float *)malloc(sizeof(float) * m * n);
  cudaMemcpy(tempmatrix, A, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) {
      float Areg = tempmatrix[col + row * n];
      printf("(%d,%d)%.3f,", row, col, Areg);
    }
    printf("\n");
  }
}

/**
 * @brief Prints m vector A on device memory. 
 */
void print_device_vector(int m, const float *A) {
  float *tempmatrix;
  tempmatrix = (float *)malloc(sizeof(float) * m);
  cudaMemcpy(tempmatrix, A, sizeof(float) * m, cudaMemcpyDeviceToHost);
  for (int row = 0; row < m; row++) {
    float Areg = tempmatrix[row];
    printf("(%d)%.3f,", row, Areg);
  }
  printf("\n");
}

void printVector(int m, const float *A, const char *name) {
  for (int i = 0; i < m; i++) {
    float Areg = A[i];
    printf("%.6f\n", Areg);
    printf("%s(%d) = %.3f\n", name, i, Areg);
  }
}