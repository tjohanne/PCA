#include "csv.cpp"
#include "include/pca.cuh"
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
  //  load data file
  std::string filename = argv[1];
  csvInfo csv = read_csv("../data/" + filename);

  //  number of principal components to find
  int ncomponents = std::stoi(argv[2]);

  //  error tolerance for eigenvectors
  const float tolerance = std::atof(argv[3]);

  //  max iterations for jacobi
  const int max_sweeps = std::stoi(argv[4]);

  //  drop irrelevant submatrices from SVD
  const int economy = std::stoi(argv[5]);

  //  print debug output
  bool verbose = false;
  if (std::stoi(argv[6]) > 0){
    verbose = true;
  }

  assert(ncomponents > 0);
  std::cout << "ncomponents " << ncomponents << "\n";
  std::cout << "tolerance " << tolerance << "\n";
  std::cout << "max_sweeps " << max_sweeps << "\n";
  std::cout << "economy " << economy << "\n";

  printf("Calling PCA with n_components %d ", ncomponents);
  printf("samples %d features %d \n", csv.rows, csv.cols);
  TimeLogger *tl = new TimeLogger(csv.rows, csv.cols, ncomponents, "../logs/" + filename);
  TimeLogger::timeLog *total_time = tl->start("Total Time");
  float_matrix_t ret = perform_pca(csv.matrix, csv.rows, csv.cols, ncomponents, economy, tolerance, max_sweeps, verbose, tl);
  tl->stop(total_time);
  printf("PCA on file %s TOTAL Time measured: %f ms.\n", filename.c_str(), total_time->time_ms);
  write_logs(tl);

  printf("Writing output matrices\n");
  //  write results to disk
  write_matrix_csv("../output/" + filename, ret.matrix, ret.rows, ret.cols);
  write_matrix_csv("../output/S_" + filename, ret.S, min(csv.cols, csv.rows), 1);
  write_matrix_csv("../output/V_" + filename, ret.V, csv.cols, csv.cols);
  free(ret.S);
  free(ret.V);
  free(ret.matrix);
  return 0;
}