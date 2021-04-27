from datetime import datetime as dt
from sklearn.decomposition import PCA
from time import perf_counter
from helpers import *

# set dataset name
# dataset_name = "face_data.csv"
# dataset_name = "mnist_784.csv"
dataset_name = "iris.csv"
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

print(f"file: {file_name}, shape {X.shape}")

if dataset_name == "face_data.csv":
    plot_faces(faces=X, method="raw", result_path=result_path, dataset_name=dataset_name)

# run, time, and plot Vanilla PCA
ncomponents = 100
SVD_METHODS = ['auto', 'randomized', 'arpack']

for method in SVD_METHODS:
    try:
        time1 = perf_counter()
        pca = PCA(n_components=None
                , copy=True
                , whiten=False
                , svd_solver=method
                , tol=1.e-9
                , iterated_power='auto'
                , random_state=42)
        X_transformed = pca.fit_transform(X)
        time2 = perf_counter()

        
        if dataset_name == "face_data.csv":
            plot_components(pca
                            , ncomponents=ncomponents
                            , result_path=result_path
                            , dataset_name=dataset_name
                            , method=method)
            # plot_faces(X_transformed, mode=f"PCA-{method}")

        print(f"X_transformed shape {X_transformed.shape}")
        print(f"sklearn method {method} - seconds: {time2-time1}")
    except Exception as e:
        print(e)
        print(f"Error - method {method}")
        pass









# recovered_images=[reconstruction(Y, C, M, h, w, i) for i in range(len(images))]
# plot_portraits(recovered_images, celebrity_names, h, w, n_row=4, n_col=4)