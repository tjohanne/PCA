import numpy as np
import sys
import os
import imageio

images_dir = '/home/gh/kernelpca/files/785_faces/train_data/'
image_files = []

for root, dirs, files in os.walk(images_dir, topdown=False):
    for name in files:
        if name.endswith(".jpg"):
            file_name = os.path.join(root, name)
            image_files.append(file_name)

images = []

for ctr, fname in enumerate(image_files):
    if ctr % 25000 == 0:
        print(ctr)
    images.append((imageio.imread(fname, as_gray=True))/255)

images = np.array(images)
nimages = images.shape[0]
images = images.reshape((nimages, -1))
np.savetxt("/home/gh/kernelpca/files/785_faces/train_faces.csv", images, delimiter=",")