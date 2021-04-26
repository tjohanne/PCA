typedef struct float_matrix {
  float* matrix;
  int rows;
  int cols;
} float_matrix_t;
float_matrix_t perform_pca(float* matrix, int M, int N, int n_components);