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

# Create a basic convolutional neural network with a sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Normalize data
fitData = fitData / 255.0
valData = valData / 255.0

# Fit the model with the training data
model.fit(fitData, fitLabels, epochs=10, validation_data=(valData, valLabels))

# Plot the loss with both fit data and validation data over epochs
pyplot.figure()

pyplot.plot(model.history.history['loss'])
pyplot.plot(model.history.history['val_loss'])

pyplot.title('Model Loss over Epochs')

pyplot.xlabel('Epoch')
pyplot.ylabel('Loss')

pyplot.legend(['Training', 'Validation'])

#Plot accuracy over epochs
pyplot.figure()

pyplot.plot(model.history.history['accuracy'])
pyplot.plot(model.history.history['val_accuracy'])

pyplot.title('Accuracy over Epochs')

pyplot.xlabel('Epoch')
pyplot.ylabel('Accuracy')

pyplot.legend(['Training', 'Validation'])

pyplot.show()