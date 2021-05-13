import numpy as np 
import pandas as pd 

datasets = ["iris"]
data_folder = "../output/"
matrices = ["alphas", "lambdas", "trans_data"]
kernels = ["LINEAR", "RBF", "POLYNOMIAL"]
cuml_files = []
sk_files = []
for dataset in datasets:
    for kernel in kernels:
        for matrix in matrices:
            cuml_files.append("CUMLSKPCA_" + dataset + "_" + kernel + "_" + matrix + ".csv")
            sk_files.append("SKKPCA_" + dataset + "_" + kernel + "_" + matrix + ".csv")
print("cuml files", cuml_files)
print("sk_files files", sk_files)
for cuml, skpca in list(zip(cuml_files, sk_files)):
        print("Checking ", cuml, skpca)
        cuml_m = np.absolute(pd.read_csv(data_folder + cuml).values)
        sk_m = np.absolute(pd.read_csv(data_folder + skpca, header=None).values)
        if "lambdas" not in cuml:
            cuml_m = cuml_m.T
        idx = zip(*np.where(~np.isclose(cuml_m, sk_m, rtol=1e-1)))
        counter = 0
        for x, y in idx:
            print("x y", x, y)
            print("MISMATCH idx", x, y, " cuml ", cuml_m[x][y], " sk_m", sk_m[x][y])
            counter += 1
            if counter > 4:
                break
        if(not np.allclose(cuml_m, sk_m, rtol=1e-1)):
            print("MISMATCH cuml", cuml, " skpca", skpca)
            print("First 20 cuml ", cuml_m[:10, :10])
            print("First 20 sk_m ", sk_m[:10, :10])

# jacobi_data = "../output/jacobi_"
# approx_data = "../output/approx_"
# result_path = "../bench/sklearn/images/"
# approx = "approx"
# jacobi = "jacobi"

# jU_file_name = jacobi_data + dataset
# jS_file_name = jacobi_data + "S_" + dataset
# jVT_file_name = jacobi_data + "V_" + dataset

# aU_file_name = approx_data + dataset
# aS_file_name = approx_data + "S_" + dataset
# aVT_file_name = approx_data + "V_" + dataset

# a_S = pd.read_csv(aS_file_name).values
# j_S = pd.read_csv(jS_file_name).values
# a_U = pd.read_csv(aU_file_name).values
# j_U = pd.read_csv(jU_file_name).values
# a_VT = pd.read_csv(aVT_file_name).values
# j_VT = pd.read_csv(jVT_file_name).values
# print(a_U[:10], j_U[:10], sep="\n=======\n")
# print('S', np.allclose(a_S, j_S, rtol=1e-02))
# print('U', np.allclose(a_U, j_U, rtol=1e-02))
# print('VT', np.allclose(a_VT, j_VT, rtol=1e-02))