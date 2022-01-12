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

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as pyplot

# Your working code here

#Load CIFAR-10 Dataset
(fitData, fitLabels), (valData, valLabels) = tf.keras.datasets.cifar10.load_data()

# Create a Sequential model with appropriate input shape for data set (32x32 image w/ 3 color channels)
model = tf.keras.Sequential([
    tf.keras.layers.Normalization(),
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

#Normalize inputs
fitData = fitData / 255.0
valData = valData / 255.0

model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# Fit the model with the training data
model.fit(fitData, fitLabels, epochs=100, validation_data=(valData, valLabels))

# Plot the loss with both fit data and validation data over epochs
print(model.history.history.keys())

pyplot.plot(model.history.history['loss'])
pyplot.plot(model.history.history['val_loss'])

pyplot.title('Model Loss over Epochs')

pyplot.xlabel('Epoch')
pyplot.ylabel('Loss')

pyplot.legend(['Training', 'Validation'])

pyplot.show()