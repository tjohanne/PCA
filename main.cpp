#include "csv.cpp"
#include "pca.cuh"

typedef struct SVD {
  float *U;
  float *S;
  float *V;
} svd_t;
svd_t perform_svd(float *A, int m, int n);

int main() {
  // csvInfo csv = read_csv("./files/iris.csv");
  // csvInfo csv = read_csv("./files/mnist_train.csv");
  csvInfo csv = read_csv("./files/face_data.csv");
  // print_csv(csv);

  int ncomponents = 2;
  printf("calling pca \n");
  perform_pca(csv.matrix, csv.rows, csv.cols, ncomponents);
  return 1;
}