#!/bin/bash

make clean && make

echo "IRIS PCA"
./cudaPca iris.csv 4

# echo "MNIST PCA"
# ./cudaPca mnist_784.csv

echo "EIGENFACES PCA"
# inputs are "file name" and ncomponents
./cudaPca face_data.csv 400
