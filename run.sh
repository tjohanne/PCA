#!/bin/bash
export PATH="/usr/local/cuda-11.2/bin:$PATH" &&
export LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH" &&
make clean && make -j 32 &&

# echo "IRIS PCA"
# ./cudaPca iris.csv 4 0

echo "MNIST PCA" &&
./cudaPca mnist_784.csv 784 0

# echo "EIGENFACES PCA"
# inputs are "file name" and ncomponents
# ./cudaPca face_data.csv 400 1
