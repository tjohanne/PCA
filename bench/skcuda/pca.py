import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
from datetime import datetime as dt
from skcuda.linalg import PCA as cuPCA

N = 100000 # TODO Is this correct M and N?
M = 1000
COMPONENTS = 2
print("Starting SKCUDA benchmark")
start_time = dt.now()
pca = cuPCA(n_components=COMPONENTS) # map the data to 4 dimensions
X = np.array([[-1.0, -1.0], [-2.0, -1.0], [-3.0, -2.0], [1.0, 1.0], [2.0, 1.0], [3.0, 2.0]])
# X = np.random.rand(N,M)
X_gpu = gpuarray.GPUArray((6,2), np.float64, order="F") # note that order="F" or a transpose is necessary. fit_transform requires row-major matrices, and column-major is the default
X_gpu.set(X) # copy data to gpu
T_gpu = pca.fit_transform(X_gpu) # calculate the principal components
print(linalg.dot(T_gpu[:,0], T_gpu[:,1]))
print("get_n_components()", pca.get_n_components())
end_time = dt.now()
elapsed=end_time-start_time

print("SKCUDA Total timeTime: %02d:%02d:%02d:%02d" % (elapsed.days, elapsed.seconds // 3600, elapsed.seconds // 60 % 60, elapsed.seconds % 60))