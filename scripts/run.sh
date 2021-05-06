#!/bin/bash
export PATH="/usr/local/cuda-11.2/bin:$PATH" &&
export LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH" &&

cd src && 
make clean && make -j 32 &&
cd ../objs &&

# echo "IRIS PCA"
# ./cudaPca iris.csv 4

DATA=mnist_784.csv
NCOMP=784
TOL=1.e-7
MAXSWEEPS=150
ECON=1
VERBOSITY=0
echo "MNIST PCA" &&
<<<<<<< HEAD:run.sh
./cudaPca mnist_784.csv 783
=======
./cudaPca $DATA $NCOMP $TOL $MAXSWEEPS $ECON $VERBOSITY

>>>>>>> 18944509130c568535235fa95c68061d9fe614ed:scripts/run.sh

# echo "EIGENFACES PCA"
# ./cudaPca face_data.csv 400
