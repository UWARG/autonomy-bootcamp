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
import matplotlib.pyplot as plt

cifar10 = tf.keras.datasets.cifar10 #loading in cifar10 dataset
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data() #extracting samples and labels for training and testing

#normalize the training and testing samples so pixel values are between 0-1
xTrain = tf.keras.utils.normalize(xTrain, axis=1)
xTest = tf.keras.utils.normalize(xTest, axis=1)

#adding input layer
model = tf.keras.models.Sequential()

#first hidden layer
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) #32 units 2D convolution layer with (3,3) kernel size
model.add(tf.keras.layers.MaxPooling2D((2, 2))) #takes max value from pool matrix of (2,2)

#second hidden layer
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu')) #64 units 2D convolution layer with (3,3) kernel
model.add(tf.keras.layers.MaxPooling2D((2, 2))) #takes max value from pool matrix of (2,2)

#third hidden layer
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu')) #64 units 2D convolution layer with (3,3) kernel
model.add(tf.keras.layers.Flatten()) #convert pixel map into 1D array
model.add(tf.keras.layers.Dense(64, activation='relu')) #fully connected layer with 64 units

#output layer for 10 outputs
model.add(tf.keras.layers.Dense(10)) #fully connected layer with 10 units for 10 outputs

#compile model using adam optimizer and crossentropy loss
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

#train model for 10 epochs on training and validation data
cifar10Model = model.fit(xTrain, yTrain, epochs=10, validation_data=(xTest, yTest)) #max accuracy about 70%

#graph loss per epoch for training and validation for 10 epoches
trainLoss = cifar10Model.history['loss']
valLoss = cifar10Model.history['val_loss']
epochs = range(1,11)
plt.plot(epochs, trainLoss, label='Training Loss')
plt.plot(epochs, valLoss, label='Validation Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
