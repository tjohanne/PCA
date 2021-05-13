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

# test = "manual"
test = "iris"

X = np.array([[1.0, 2.0, 5.0], [4.0, 2.0, 1.0]]).T

if test == "iris":
    print("Loading iris dataset")
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
print("Shape", X.shape)
time_init_pca = dt.now()
kpca = KernelPCA(n_components=None
                 , kernel='linear'
                 , gamma=None
                 , degree=3
                 , coef0=1
                 , kernel_params=None
                 , alpha=1.0
                 , fit_inverse_transform=False
                 , eigen_solver='auto'
                 , tol=0
                 , max_iter=None
                 , remove_zero_eig=False
                 , random_state=None
                 , copy_X=True
                 , n_jobs=-1)
kpca.fit_transform(X)

# print("SKLEARN KPCA Time for fit_transform {}ms".format((dt.now() - time_fit_transform).microseconds / 1000))
print("SKLEARN KPCA Total time for fit_transform {}ms".format((dt.now() - time_init_pca).microseconds / 1000))

