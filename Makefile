EXECUTABLE := cudaPca
CU_FILES   := pca.cu eigensolver.cu
CU_DEPS    :=
CC_FILES   := main.cpp


all: $(EXECUTABLE)

###########################################################

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall
# Add in cuda library?
LDFLAGS=-L/usr/local/cuda-11.2/lib64/ -lcudart -lcublas -lcusolver
NVCC=nvcc
NVCCFLAGS= -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc-8 
OBJS= $(OBJDIR)/main.o $(OBJDIR)/pca.o $(OBJDIR)/eigensolver.o

.PHONY: dirs clean

all: $(EXECUTABLE) clang-format

clang-format: 
	clang-format -i *.cpp *.cu

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)



$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
