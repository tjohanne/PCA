from datetime import datetime as dt
from sklearn.decomposition import PCA
from time import perf_counter
from helpers import *

# set dataset name
data_file_name = "face_data.csv"
dataset_name = "Eigenfaces"
data_dir = "/home/gh/kernelpca/output/"
result_path = "/home/gh/kernelpca/bench/sklearn/images/"
U_file_name = data_dir + data_file_name
S_file_name = data_dir + "S_" + data_file_name
VT_file_name = data_dir + "V_" + data_file_name

# read data
U = pd.read_csv(U_file_name).values
S = pd.read_csv(S_file_name).values
VT = pd.read_csv(VT_file_name).values

plot_components(components=VT
            , ncomponents=400
            , result_path=result_path
            , dataset_name=dataset_name
            , method="cuda_jacobi")





# recovered_images=[reconstruction(Y, C, M, h, w, i) for i in range(len(images))]
# plot_portraits(recovered_images, celebrity_names, h, w, n_row=4, n_col=4)


