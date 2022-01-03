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

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Your working code here

# CONSTANTS
BATCH_SIZE = 32
EPOCHS = 10

# Distributing data into variables
dataSet = tf.keras.datasets.cifar10
(xTrain, yTrain), (xTest, yTest) = dataSet.load_data()

# Normalizing data so that the data is organized and on a common scale. 
# This helps the model converge quicker.
xTrain = tf.keras.utils.normalize(xTrain, axis = 1)
xTest = tf.keras.utils.normalize(xTest, axis = 1)


# Input layer, Convolutional Layer with 128 Filters and Pooling Layer
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(128, (3,3), input_shape = xTrain.shape[1:]))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.1))

# Second layer, Convolutional Layer with 64 Filters and Pooling Layer
model.add(tf.keras.layers.Conv2D(64, (3,3)))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.1))

# Third layer, Dense Layer with 128 Filters
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.Activation("relu"))

# Output layer, Dense Layer with softmax activation
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation("softmax"))

# Compile model
model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# Fit the model and store it in a variable
historyTrain = model.fit(xTrain, yTrain, batch_size = BATCH_SIZE, epochs = EPOCHS)

# Plots and Testing Model Accuracy

lossTrain = historyTrain.history['loss']
accTrain = historyTrain.history['accuracy']
epochs = range(1, EPOCHS+1)


plt.plot(epochs, lossTrain, 'g', label='Training Loss Over Time')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, accTrain, 'b', label='Training Accuracy Over Time')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

valLoss, valAcc = model.evaluate(xTest, yTest)
print("Validation loss: ", valLoss, "\nValidation Accuracy: ", valAcc*100, "%")

