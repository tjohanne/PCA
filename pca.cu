#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cublas_v2.h"
#include "svd.cuh"
#include <cuda_runtime.h>
#ifndef min
#define min(a, b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a, b) ((a > b) ? a : b)
#endif
//
// Nearly minimal CUDA example.
// Compile with:
//
// nvcc -o example example.cu
//

//
// A function marked __global__
// runs on the GPU but can be called from
// the CPU.
//
// This function multiplies the elements of an array
// of ints by 2.
//
// The entire computation can be thought of as running
// with one thread per array element with blockIdx.x
// identifying the thread.
//
// The comparison i<N is because often it isn't convenient
// to have an exact 1-1 correspondence between threads
// and array elements. Not strictly necessary here.
//
// Note how we're mixing GPU and CPU code in the same source
// file. An alternative way to use CUDA is to keep
// C/C++ code separate from CUDA code and dynamically
// compile and load the CUDA code at runtime, a little
// like how you compile and load OpenGL shaders from
// C/C++ code.
//

void perform_pca(float *matrix, int M, int N) {
  matrix[0] = 4.0;
  matrix[1] = 0.0;
  matrix[2] = 3.0;
  matrix[3] = -5.0;
  svd_t svd = perform_svd(matrix, 2, 2);
}