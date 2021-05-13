// main.cpp
#include <stdio.h>
#include <iostream>
#include <cuml/decomposition/params.hpp>
#include <cuml/decomposition/kpca.hpp>
#include <cuml/matrix/kernelparams.h>

#include <cuda_runtime.h>

#include <raft/handle.hpp>
#include <raft/mr/device/allocator.hpp>
#include <raft/cudart_utils.h>
#include <raft/linalg/cublas_wrappers.h>

#include <raft/cuda_utils.cuh>
#include "../../src/csv.cpp"

int main(int argc, char *argv[]) {
  cudaStream_t stream = CUDA_CHECK(cudaStreamCreate(&stream));
  raft::handle_t handle;
  handle.set_stream(stream);
  std::cout << "Here\n";
  //  load data file
  std::string filename = argv[1];
  std::cout << "Here\n" << " ../../data/" + filename;
  csvInfo csv = read_csv("../../data/" + filename);
  std::cout << "Here\n";

  //  number of principal components to find
  int n_components = std::stoi(argv[2]);
  std::string kernel = argv[3];

  ML::paramsKPCA prms;
  float* data;
  float* trans_data;
  int len = csv.rows * csv.cols;
  raft::allocate(data, len);
  raft::allocate(trans_data, csv.rows * n_components); // transformed data

  std::vector<T> data_h = {1.0, 2.0, 5.0, 4.0, 2.0, 1.0};
  data_h.resize(len);
  raft::update_device(data, csv.matrix, len, stream);
  prms.n_cols = 2;
  prms.n_rows = 3;
  prms.n_components = 4;
  prms.kernel = MLCommon::Matrix::KernelParams{MLCommon::Matrix::LINEAR, 0, 0.0, 0.0};
  if(kernel == "POLYNOMIAL") {
    prms.kernel.kernel = MLCommon::Matrix::POLYNOMIAL;
  }
  else if(kernel == "RBF") {
      prms.kernel.kernel = MLCommon::Matrix::RBF;
  }
  else if(kernel == "TANH") {
      prms.kernel.kernel = MLCommon::Matrix::TANH;
  }
  else if(kernel != "LINEAR") {
    std::cout << kernel << " is not a valid kernel type " << std::endl;
    exit(1);
  }
  std::cout << "prms.n_cols " << prms.n_cols << std::endl;
  std::cout << "kernel " << prms.kernel.coef0 << std::endl;
  std::cout << "kernel " << prms.kernel.kernel << std::endl;
  return 0;
}
