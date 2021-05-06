#include "csv.cpp"
#include "pca.cuh"
#include <chrono>
#include <assert.h>

#ifndef min
#define min(a, b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a, b) ((a > b) ? a : b)
#endif

typedef struct SVD {
  float *U;
  float *S;
  float *V;
} svd_t;

svd_t perform_svd(float *A, int m, int n);

int main(int argc, const char *argv[]) {
  
  //  parse input arguments
  std::string filename = argv[1];
  int ncomponents = std::stoi(argv[2]);
  bool write_s_v = (bool) std::stoi(argv[3]);
  assert(ncomponents > 0);

  //  load data
  csvInfo csv = read_csv("./files/" + filename);
  printf("Read CVS with M %d N %d \n", csv.rows, csv.cols);

  //  call PCA 
  const float tolerance = 1.e-9;
  const int max_sweeps = 100;
  const int economy = 1;
  bool verbose = false;

  printf("Calling PCA with n_components %d \n", ncomponents);
  auto begin = std::chrono::high_resolution_clock::now();
  float_matrix_t ret = perform_pca(csv.matrix, csv.rows, csv.cols, ncomponents, economy, tolerance, max_sweeps, verbose, write_s_v);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  printf("TOTAL Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);
  printf("Writing to disk. Write S and V also ?%d\n", write_s_v);
  //  write results to disk
  write_matrix_csv("./output/" + filename, ret.matrix, ret.rows, ret.cols);
  if(write_s_v) {
    write_matrix_csv("./output/S_" + filename, ret.S, min(csv.cols, csv.rows), 1);
    write_matrix_csv("./output/V_" + filename, ret.V, csv.rows, csv.cols);
  }
  return 1;
}