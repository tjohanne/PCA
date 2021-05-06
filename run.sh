#!/bin/bash
export PATH="/usr/local/cuda-11.2/bin:$PATH" &&
export LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH" &&
cd src && 
make clean && make -j 32 &&
cd ../objs &&

# echo "IRIS PCA"
# ./cudaPca iris.csv 4

echo "MNIST PCA" &&
./cudaPca mnist_784.csv 783

# echo "EIGENFACES PCA"
# ./cudaPca face_data.csv 400
