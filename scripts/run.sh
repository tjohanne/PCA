#!/bin/bash
export PATH="/usr/local/cuda-11.2/bin:$PATH" &&
export LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH" &&

cd ../src && 
make clean && make -j 32 &&
cd ../objs &&

# echo "IRIS PCA"
# ./cudaPca iris.csv 4

DATA=mnist_784.csv
NCOMP=784
# DATA=mnist_784.csv
# NCOMP=784
TOL=1.e-3
MAXSWEEPS=15
ECON=1
VERBOSITY=0
#jacobi or approx solver
SOLVER=jacobi
# SOLVER=jacobi
echo "mnist 784" &&
./cudaPca $DATA $NCOMP $TOL $MAXSWEEPS $ECON $VERBOSITY $SOLVER

# echo "EIGENFACES PCA"
# ./cudaPca face_data.csv 400
