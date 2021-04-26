import numpy as np
from datetime import datetime as dt
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt
from time import perf_counter
import helpers



dataset_name = "eigenfaces"
file_name = "/home/gh/kernelpca/files/face_data.csv"
result_path = "/home/gh/kernelpca/bench/sklearn/images/"
X = pd.read_csv(file_name).drop('target', axis=1).values
print(f"file: {file_name}, shape {X.shape}")
plot_faces(X, mode="input")




# recovered_images=[reconstruction(Y, C, M, h, w, i) for i in range(len(images))]
# plot_portraits(recovered_images, celebrity_names, h, w, n_row=4, n_col=4)



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
                , tol=0.0
                , iterated_power='auto'
                , random_state=42)
        X_transformed = pca.fit_transform(X)
        time2 = perf_counter()
        
        plot_components(pca, ncomponents=ncomponents)
        # plot_faces(X_transformed, mode=f"PCA-{method}")

        print(f"X_transformed shape {X_transformed.shape}")
        print(f"scikit-learn method {method} - seconds: {time2-time1}")
    except Exception as e:
        print(e)
        print(f"Error - method {method}")
        pass








