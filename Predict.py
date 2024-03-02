# Importing libraries
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.models import load_model

# Categorising the data as Healthy and Diseased leaves
Categories = ["Healthy", "Diseased"]


def preprocess_img(image_path, img_shape=(100, 100)):
    image = cv.imread(image_path)
    image = cv.resize(image, img_shape)
    return image.reshape(-1, img_shape[0], img_shape[1], 3)


# Getting the path to the Model and Image
model_path = str(input('Enter the relative path to the Model : '))
image_path = str(input('Enter the relative path to the Image : '))
model = load_model(model_path)
prediction = model.predict([preprocess_img(image_path)])

if int(prediction) == 0:
    print(Categories[0], " leaf")

else:
    print(Categories[1], " leaf")
