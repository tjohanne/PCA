#!/bin/bash
export PATH="/usr/local/cuda-11.2/bin:$PATH" &&
export LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH" &&

cd ../src && 
make clean && make -j 32 &&
cd ../objs &&

# echo "IRIS PCA"
# ./cudaPca iris.csv 4 0

DATA=mnist_784.csv
NCOMP=784
TOL=1.e-7
MAXSWEEPS=150
ECON=1
VERBOSITY=0
echo "MNIST PCA" &&
./cudaPca $DATA $NCOMP $TOL $MAXSWEEPS $ECON $VERBOSITY


# echo "EIGENFACES PCA"
# inputs are "file name" and ncomponents
# ./cudaPca face_data.csv 400 1
