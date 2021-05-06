import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import pandas as pd
import skcuda.linalg as linalg
from datetime import datetime as dt
from skcuda.linalg import PCA as cuPCA

dataset_name = "mnist_784.csv"
data_dir = "../../files/"
file_name = data_dir + dataset_name
X = pd.read_csv(file_name)
if 'class' in X:
    X = X.drop('class', axis=1)
if 'target' in X:
    X = X.drop('target', axis=1)
if 'variety' in X:
    X = X.drop('variety', axis=1)
X = np.array(X.values, dtype="float64")
samples, features = X.shape
n_components = features
print("SKCUDA Running PCA with {} features, {} samples, and {} n_components on dataset {}".format(X.shape[1], X.shape[0], n_components, dataset_name))
time_init_pca = dt.now()
pca = cuPCA(n_components=n_components) # map the data to 4 dimensions
X_gpu = gpuarray.GPUArray((samples,features), np.float64, order="F") # note that order="F" or a transpose is necessary. fit_transform requires row-major matrices, and column-major is the default
X_gpu.set(X) # copy data to gpu
T_gpu = pca.fit_transform(X_gpu) # calculate the principal components
print(linalg.dot(T_gpu[:,0], T_gpu[:,1]))
print("get_n_components()", pca.get_n_components())

print("SKCUDA Total time {}ms".format((dt.now() - time_init_pca).microseconds / 1000))