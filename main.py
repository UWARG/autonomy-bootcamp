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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, utils, layers, models

# Your working code here

# Load the CIFAR10 Dataset
(trainX, trainY), (testX, testY) = datasets.cifar10.load_data()

# Converting the labels into a new binary variable for the category
trainY = utils.to_categorical(trainY)
testY = utils.to_categorical(testY)

# Convert from integer to float
trainX = trainX.astype('float32')
testX = testX.astype('float32')

# Normalize the images to the range of 0 to 1
trainX = trainX / 255.0
testX = testX / 255.0

# Constant Variables
NUM_OF_CLASSES = 10
HEIGHT = 32
WIDTH = 32
EPOCHS = 25
BATCH_SIZE = 64

# Next steps are initializing and developing the CNN model. The feature extraction part of the model is to be
# created using a double convolutional layer, followed by a batch normalization layer and then a max pooling
# layer. It will end with a dropout layer. This will be repeated 3 times with the first convolutional layer to
# have 32 nodes, the second convolutional layer to have 64 nodes, and the third convolutional layer to have 128
# nodes. The dropout layer will increase by 0.1 every time.

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(HEIGHT, WIDTH, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Dropout(0.4))

# The model will then be flattened to enter the last hidden layer. The last hidden layer has 128 nodes, followed
# by a Batch Normalization layer, a Dropout layer, then the output layer with 10 nodes and the number of classes
# in the CIFAR10 dataset is 10

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model using the adam optimizer, categorical cross entropy loss for multi-class classification and
# monitor accuracy value

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(trainX, trainY, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(testX, testY))

# Plot of both training and validation losses over epochs
plt.plot(history.history['loss'], color='blue', label='Training Loss')
plt.plot(history.history['val_loss'], color='red', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.show()

# Plot of both training and validation accuracy over epochs
plt.plot(history.history['accuracy'], color='blue', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], color='red', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Evaluating the model with the test images to produce the final test accuracy value
_, testAccuracy = model.evaluate(testX, testY)
print('\nTEST ACCURACY: ' + '%.3f' % (testAccuracy * 100.0))
x