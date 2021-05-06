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
  
  //  parse input arguments
  std::string filename = argv[1];
  int ncomponents = std::stoi(argv[2]);
  const float tolerance = std::atof(argv[3]);
  const int max_sweeps = std::stoi(argv[4]);
  const int economy = std::stoi(argv[5]);
  bool verbose = false;
  if (std::stoi(argv[6]) > 0){
    verbose = true;
  }

  assert(ncomponents > 0);
  std::cout << "ncomponents " << ncomponents << "\n";
  std::cout << "tolerance " << tolerance << "\n";
  std::cout << "max_sweeps " << max_sweeps << "\n";
  std::cout << "economy " << economy << "\n";

  //  load data
  csvInfo csv = read_csv("../data/" + filename);

  //  call PCA 
  const float tolerance = 1.e-9;
  const int max_sweeps = 100;
  const int economy = 1;
  bool verbose = false;

  printf("Calling PCA with n_components %d ", ncomponents);
  printf("samples %d features %d \n", csv.rows, csv.cols);
  TimeLogger *tl = new TimeLogger(csv.rows, csv.cols, ncomponents, "../logs/" + filename);
  TimeLogger::timeLog *total_time = tl->start("Total Time");
  float_matrix_t ret = perform_pca(csv.matrix, csv.rows, csv.cols, ncomponents, economy, tolerance, max_sweeps, verbose, tl);
  tl->stop(total_time);
  printf("%s TOTAL Time measured: %.3f ms.\n", total_time->time_ms);
  write_logs(tl);

  
  //  write results to disk
  write_matrix_csv("../output/" + filename, ret.matrix, ret.rows, ret.cols);
  write_matrix_csv("../output/S_" + filename, ret.S, min(csv.cols, csv.rows), 1);
  write_matrix_csv("../output/V_" + filename, ret.V, csv.cols, csv.cols);
  free(ret.S);
  free(ret.V);
  free(ret.matrix);
  return 0;
}