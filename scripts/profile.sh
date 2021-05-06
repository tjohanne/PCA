#!/bin/bash

export PATH="/usr/local/cuda-11.2/bin:$PATH" &&
export LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH" &&

cd src && 
make clean && make -j 32 &&
cd ../objs &&

echo "MNIST PCA" &&


ncu ./cudaPca mnist_784.csv 784 1
