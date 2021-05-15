import sklearn
import numpy as np
from sklearn.decomposition import KernelPCA
from scipy import linalg
from scipy.sparse.linalg import eigsh
from sklearn.utils.extmath import svd_flip
from sklearn.utils.validation import check_is_fitted, _check_psd_eigenvalues
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn import datasets
from datetime import datetime as dt
import pandas as pd

# test = "manual"
data_folder = "../../data/"
dataset = "mnist_784"
test = dataset + ".csv"
# top_rows_arr = [100, 500, 1000, 2000, 5000, 10000]
top_rows_arr = [15000, 20000, 30000, 40000, 50000]
for top_rows in top_rows_arr:
    X = pd.read_csv(data_folder + test, nrows=top_rows)
    X.to_csv(data_folder + dataset + "_" + str(top_rows) + ".csv", index=False)
# out_folder = "../../output/SKKPCA_" + test + "_"
# log_file = "../../logs/SKKPCA_" + test + ".csv"
# rows, cols = X.shape