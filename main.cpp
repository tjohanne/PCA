#include "csv.cpp"
#include "pca.cuh"

typedef struct SVD {
  float *U;
  float *S;
  float *V;
} svd_t;
svd_t perform_svd(float *A, int m, int n);

int main() {
  csvInfo csv = read_csv("./files/iris.csv");
  perform_pca(csv.matrix, csv.rows, csv.cols);
  // print_csv(csv); // Can uncomment if you want to see output.
  return 1;
}