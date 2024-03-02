#-------------------------------------------------------------------------------------------
# From the experiments conducted it has been observed that This Architecture performs the best among others.
# 3 Hidden layers with (32,64,32) neurons respectively.
#-------------------------------------------------------------------------------------------


# Importing Libraries

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Getting the values from the pickle files

pickle_in = open("Pickle/X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("Pickle/y.pickle", "rb")
Y = pickle.load(pickle_in)

# Changing the range of the Pixel values from 0-255 to 0-1 for easier calculation

X = X/255.0

model = Sequential()

# First Hidden layer

model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Hidden layer

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Hidden layer

model.add(Flatten()) 
model.add(Dense(32))
model.add(Activation('relu'))

# Output layer

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X, Y, batch_size=32, epochs=6, validation_split=0.3)

model.save('Best.model')
