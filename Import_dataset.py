# Importing the libraries 
import numpy as np
import os
import cv2 as cv
import random
import pickle
from tqdm import tqdm

# Relative Path 
Directory = "Dataset"

# Categorising the data as Healthy and Diseased leaves
Categories = ["Healthy", "Diseased"]
img_shape = (100, 100)
X = []
Y = []
training_data = []


def create_training_data():
    for i in Categories:

        path = os.path.join(Directory, i)
        class_num = Categories.index(i)

        for j in tqdm(os.listdir(path)):
            try:
                img_array = cv.imread(os.path.join(path, j))
                new_array = cv.resize(img_array, img_shape)
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()
random.shuffle(training_data)
for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, img_shape[0], img_shape[1], 3)
Y = np.array(Y)

# Getting the data out as Pickle file, to be used for training
pickle_out = open("Pickle/X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Pickle/y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()
