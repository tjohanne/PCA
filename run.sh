#!/bin/bash

make clean && make

echo "IRIS PCA"
./cudaPca iris.csv

echo "MNIST PCA"
./cudaPca mnist_784.csv

echo "EIGENFACES PCA"
./cudaPca face_data.csv
