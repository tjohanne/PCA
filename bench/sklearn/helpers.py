import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plot_faces(faces, method, result_path, dataset_name):
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(faces[i].reshape(64, 64), cmap=plt.cm.bone)

    plt.title("Eigenfaces - Raw Inputs")
    plt.savefig(f'{result_path}{dataset_name}-{method}.png')
    plt.show()

def plot_components(components, ncomponents, result_path, dataset_name, method):
    fig = plt.figure(figsize=(16, 6))
    for i in range(30):
        ax = fig.add_subplot(3, 10, i + 1, xticks=[], yticks=[])
        ax.set_title(f"Comp. {i}")
        ax.imshow(components[i].reshape((64, 64)), cmap=plt.cm.bone)
    
    plt.savefig(f'{result_path}{dataset_name}-PCA-components-{ncomponents}-method-{method}.png')


def reconstruction(Y, C, M, h, w, image_index):
    n_samples, n_features = Y.shape
    weights = np.dot(Y, C.T)
    centered_vector=np.dot(weights[image_index, :], C)
    recovered_image=(M+centered_vector).reshape(h, w)
    return recovered_image
