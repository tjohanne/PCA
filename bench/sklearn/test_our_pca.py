from datetime import datetime as dt
from sklearn.decomposition import PCA
from time import perf_counter
from helpers import *

# set dataset name
dataset_name = "face_data.csv"
data_dir = "/home/gh/kernelpca/output/"
result_path = "/home/gh/kernelpca/bench/sklearn/images/"
file_name = data_dir + dataset_name

# read data
X = pd.read_csv(file_name)
if 'class' in X:
    X = X.drop('class', axis=1)
if 'target' in X:
    X = X.drop('target', axis=1)
if 'variety' in X:
    X = X.drop('variety', axis=1)
X = X.values

our_X = X


print(our_X[0, :5])

print(f"file: {file_name}, shape {X.shape}")




# set dataset name
dataset_name = "face_data.csv"
data_dir = "/home/gh/kernelpca/files/"
result_path = "/home/gh/kernelpca/bench/sklearn/images/"
file_name = data_dir + dataset_name


# read data
X = pd.read_csv(file_name)
if 'class' in X:
    X = X.drop('class', axis=1)
if 'target' in X:
    X = X.drop('target', axis=1)
if 'variety' in X:
    X = X.drop('variety', axis=1)
X = X.values

sklearn_X = X

pca = PCA(n_components=None
        , copy=True
        , whiten=False
        , svd_solver='auto'
        , tol=1.e-9
        , iterated_power='auto'
        , random_state=42)
X_transformed = pca.fit_transform(X)



print(X_transformed[0, :5])
# recovered_images=[reconstruction(Y, C, M, h, w, i) for i in range(len(images))]
# plot_portraits(recovered_images, celebrity_names, h, w, n_row=4, n_col=4)