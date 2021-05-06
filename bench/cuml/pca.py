from datetime import datetime as dt
from cuml.decomposition import PCA
import numpy as np
from time import perf_counter
import pandas as pd

# set dataset name
dataset_name = "mnist_784.csv"
data_dir = "../../files/"
file_name = data_dir + dataset_name

# read data
X = pd.read_csv(file_name)
X = X.apply(pd.to_numeric, errors='coerce')
if 'class' in X:
    X = X.drop('class', axis=1)
if 'target' in X:
    X = X.drop('target', axis=1)
if 'variety' in X:
    X = X.drop('variety', axis=1)

X = X.values
X = X.astype("float32")
sklearn_X = X
n_components = min(X.shape)
time_init_pca = dt.now()
print("CUML Running PCA with {} features, {} samples, and {} n_components on dataset {}".format(X.shape[1], X.shape[0], n_components, dataset_name))
pca = PCA(n_components=min(X.shape)
        , copy=True
        , whiten=False
        , svd_solver='jacobi'
        , tol=1.e-9
        , iterated_power=15
        , random_state=42)
time_fit_transform = dt.now()
X_transformed = pca.fit_transform(X)
print("CUML Time for transform {}ms".format((dt.now() - time_fit_transform).microseconds / 1000))
print("CUML Total time {}ms".format((dt.now() - time_init_pca).microseconds / 1000))