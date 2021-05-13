export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib" &&
nvcc \
       main.cpp \
       -o kpcBench \
       "-L${CONDA_PREFIX}/lib" \
       "-L/home/tomas/618/raft" \
       "-I${CONDA_PREFIX}/include" \
       "-I${CONDA_PREFIX}/include/cuml/raft" \
       -lcuml++ &&
DATA=iris.csv &&
NCOMP=4 &&
# LINEAR RBF POLYNOMIAL 
KERNEL=LINEAR && 
./kpcBench $DATA $NCOMP $KERNEL