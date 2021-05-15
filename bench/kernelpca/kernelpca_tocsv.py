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
test = "mnist_784_2000"
df = pd.read_csv(data_folder + test + ".csv", nrows=1) # read just first line for columns
columns = df.columns.tolist() # get the columns
cols_to_use = columns[:len(columns)-1] # drop the last one
X = pd.read_csv(data_folder + test + ".csv", usecols=cols_to_use)
X = X.values
out_folder = "../../output/SKKPCA_" + test + "_"
log_file = "../../logs/SKKPCA_" + test + ".csv"
# if test == "iris":
#     print("Loading iris dataset")
#     iris = datasets.load_iris()
#     X = iris.data
#     y = iris.target
print("Shape", X.shape)
rows, cols = X.shape
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
# kernels = kernels[:2]
log = ["Function Name,Features,Samples,N Components,Time"]

for i in range(len(kernels)):
    input_kernel, out_kernel = kernels[i]
    degree, gamma, coef0 = kernel_params[i]
    time_init_pca = dt.now()
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
    print("kernel",input_kernel,"lambdas", lambdas)
    log.append("kpca.fit_transform {},{},{},{},{}".format(input_kernel, cols, rows, rows, (dt.now() - time_init_pca)))
    print("SKLEARN KPCA kernel {} Total time for fit_transform {}ms".format(input_kernel, (dt.now() - time_init_pca)))
    print(input_kernel, "trans_out", trans_out.shape, "alphas", alphas.shape, "lambdas", lambdas.shape)
    # np.savetxt(out_folder + out_kernel + "_trans_data.csv", trans_out, delimiter=",")
    # np.savetxt(out_folder + out_kernel + "_alphas.csv", alphas, delimiter=",")
    # np.savetxt(out_folder + out_kernel + "_lambdas.csv", lambdas, delimiter=",")


# print("SKLEARN KPCA Time for fit_transform {}ms".format((dt.now() - time_fit_transform).microseconds / 1000))

with open(log_file, 'w') as filehandle:
    for listitem in log:
        filehandle.write('%s\n' % listitem)