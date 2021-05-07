import numpy as np 
import pandas as pd 

dataset = "iris.csv"
jacobi_data = "../output/jacobi_"
approx_data = "../output/approx_"
result_path = "../bench/sklearn/images/"
approx = "approx"
jacobi = "jacobi"

jU_file_name = jacobi_data + dataset
jS_file_name = jacobi_data + "S_" + dataset
jVT_file_name = jacobi_data + "V_" + dataset

aU_file_name = approx_data + dataset
aS_file_name = approx_data + "S_" + dataset
aVT_file_name = approx_data + "V_" + dataset

a_S = pd.read_csv(aS_file_name).values
j_S = pd.read_csv(jS_file_name).values
a_U = pd.read_csv(aU_file_name).values
j_U = pd.read_csv(jU_file_name).values
a_VT = pd.read_csv(aVT_file_name).values
j_VT = pd.read_csv(jVT_file_name).values
print(a_U[:10], j_U[:10], sep="\n=======\n")
print('S', np.allclose(a_S, j_S, rtol=1e-02))
print('U', np.allclose(a_U, j_U, rtol=1e-02))
print('VT', np.allclose(a_VT, j_VT, rtol=1e-02))