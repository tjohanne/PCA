#!/bin/bash
export PATH="/usr/local/cuda-11.2/bin:$PATH" &&
export LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH" &&

cd ../src && 
make clean && make -j 32 &&
cd ../objs &&


DATA=mnist_784.csv
NCOMP=784
TOL=1.e-3
MAXSWEEPS=15
ECON=1
VERBOSITY=0
SOLVER=jacobi
echo "MNIST PCA"
ncu -o profile --target-processes all --details-all --print-summary per-gpu --replay-mode application ./cudaPca $DATA $NCOMP $TOL $MAXSWEEPS $ECON $VERBOSITY $SOLVER




