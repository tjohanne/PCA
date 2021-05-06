typedef struct float_matrix {
  float* matrix;
  float* S;
  float* V;
  int rows;
  int cols;
} float_matrix_t;
float_matrix_t perform_pca(float* matrix, int M, int N, int n_components, const int econ, const float tol,
                            const int max_sweeps, const bool verbose, const bool include_s_v);