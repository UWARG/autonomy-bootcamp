"""
This is a starter file to get you going. You may also include other files if you feel it's necessary.

Make sure to follow the code convention described here:
https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions

Hints:
* The internet is your friend! Don't be afraid to search for tutorials/intros/etc.
* We suggest using a convolutional neural network.
* TensorFlow Keras has the CIFAR-10 dataset as a module, so you don't need to manually download and unpack it.
"""

# Import whatever libraries/modules you need

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

# Your working code here
# Authored by Jerry Bi

# Loading in the training data
(xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()

# Normalize the data for our model
xTrain = tf.keras.utils.normalize(xTrain, axis=1)
xTest = tf.keras.utils.normalize(xTest, axis=1)

# initialize the model
model = Sequential()

# first layer of model uses 2x64 convolution
model.add(Conv2D(32, (3, 3), input_shape=xTrain.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1)) # attempts to improve validation loss

# second layer of model uses 2x64 convolution
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))  # attempts to improve validation loss

# third layer of model uses 2x64 convolution
model.add(Flatten())  # 2D data must be flattened for Dense layers
model.add(Dense(64))
model.add(Activation("relu"))

# output layer
model.add(Dense(10))
model.add(Activation("softmax"))

# run/train the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(xTrain, yTrain, batch_size=32, epochs=10, validation_data=(xTest, yTest))

# graph the loss and validation loss
loss = history.history['loss']
valLoss = history.history['val_loss']

plt.plot(loss, 'g', label='loss')
plt.plot(valLoss, 'b', label='validation loss')
plt.title('Loss and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
