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
out_folder = "../../output/SKKPCA_" + test + "_"
if test == "iris":
    print("Loading iris dataset")
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
print("Shape", X.shape)
# KernelType kernel;  //!< Type of the kernel function
#   int degree;         //!< Degree of polynomial kernel (ignored by others)
#   double gamma;       //!< multiplier in the
#   double coef0; 
#     prms.kernel = MLCommon::Matrix::KernelParams{Matrix::LINEAR, 0, 0, 0};
    # prms.kernel = MLCommon::Matrix::KernelParams{Matrix::RBF, 0, 1.0/2.0f, 0};
#     prms.kernel = MLCommon::Matrix::KernelParams{Matrix::POLYNOMIAL, 3, 1.0/2.0f, 1};
kernel_params = [(0,None,0), (0, None, 0), (3, None, 1)]
matrices = ["alphas", "lambdas", "trans_data"]
kernels = [("linear", "LINEAR"), ("rbf", "RBF"), ("poly", "POLYNOMIAL")]
for i in reversed(range(len(kernels))):
    input_kernel, out_kernel = kernels[i]
    degree, gamma, coef0 = kernel_params[i]
    kpca = KernelPCA(n_components=None
                    , kernel=input_kernel
                    , gamma=gamma
                    , degree=degree
                    , coef0=coef0
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
                    
    trans_out = kpca.fit_transform(X)
    alphas = kpca.alphas_
    lambdas = kpca.lambdas_
    print(input_kernel, "trans_out", trans_out.shape, "alphas", alphas.shape, "lambdas", lambdas.shape)
    np.savetxt(out_folder + out_kernel + "_trans_data.csv", trans_out, delimiter=",")
    np.savetxt(out_folder + out_kernel + "_alphas.csv", alphas, delimiter=",")
    np.savetxt(out_folder + out_kernel + "_lambdas.csv", lambdas, delimiter=",")

# print("SKLEARN KPCA Time for fit_transform {}ms".format((dt.now() - time_fit_transform).microseconds / 1000))

