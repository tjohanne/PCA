#include "csv.cpp"
#include "pca.cuh"
#include <chrono>

typedef struct SVD {
  float *U;
  float *S;
  float *V;
} svd_t;

svd_t perform_svd(float *A, int m, int n);

int main(int argc, const char *argv[]) {
  std::string filename = argv[1];
  csvInfo csv = read_csv("./files/" + filename);
  printf("Read CVS with M %d N %d \n", csv.rows, csv.cols);
  int ncomponents = 2;
  const float tolerance = 1.e-9;
  const int max_sweeps = 250;
  const int economy = 1;
  bool verbose = false;
  printf("Calling PCA with n_components %d \n", ncomponents);
  auto begin = std::chrono::high_resolution_clock::now();
  float_matrix_t ret = perform_pca(csv.matrix, csv.rows, csv.cols, ncomponents, economy, tolerance, max_sweeps, verbose);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);

  write_matrix_csv("./output/" + filename, ret.matrix, ret.rows, ret.cols);
  return 1;
}