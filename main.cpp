#include "csv.cpp"

void cuda_example();
int cublas_example();
int eigensolver_example();
int main() {
  csvInfo csv = read_csv("./files/iris.csv");
  // print_csv(csv); // Can uncomment if you want to see output.
  cublas_example();
  eigensolver_example();
  return 1;
}