#include "csv.cpp"
#include "pca.cuh"

typedef struct SVD {
  float *U;
  float *S;
  float *V;
} svd_t;

svd_t perform_svd(float *A, int m, int n);

int main() {
  // std::string filename = "iris.csv";
  // std::string filename = "mnist_784.csv";
  std::string filename = "face_data.csv";
  // csvInfo csv = read_csv("./files/");
  // csvInfo csv = read_csv("./files/");
  csvInfo csv = read_csv("./files/" + filename);
  // print_csv(csv);
  printf("Read CVS with M %d N %d \n", csv.rows, csv.cols);

  int ncomponents = 2;
  printf("calling pca \n");
  float_matrix_t ret = perform_pca(csv.matrix, csv.rows, csv.cols, ncomponents);
  write_matrix_csv("./output/" + filename, ret.matrix, ret.rows, ret.cols);
  return 1;
}