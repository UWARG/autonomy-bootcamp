'''
    FINAL CIFAR-10 IMAGE CLASSIFIER FOR COMPUTER VISION BOOTCAMP

    Authored by: Bassel Al-Omari
    Date: 7-Jan-2021
'''

import pickle
import numpy as np
from matplotlib import pyplot

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Used to extract information from CIFAR-10 Files
# Copied from CIFAR-10 Website
def unpickle(file):   
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
						
# When downloading from the CIFAR-10 Website, data came in 6 batches:
#     1 Testing Batch
#     5 Training Batches, which I have here concatenated all into one for ease of use

trainLabels = unpickle("CIFAR10/data_batch_1")[b'labels']
trainData = unpickle("CIFAR10/data_batch_1")[b'data']

for i in range(2,6):
    trainLabels += unpickle(f"CIFAR10/data_batch_{i}")[b'labels']
    trainData = np.concatenate((trainData, unpickle(f"CIFAR10/data_batch_{i}")[b'data']))

testLabels = unpickle("CIFAR10/test_batch")[b'labels']
testData = unpickle("CIFAR10/test_batch")[b'data']

# Dividing all values by 255.0 to normalize data
# Resizing data from a 1D to a 32x32x3 Matrix (32x32 corresponding to the size of the 
# image, and 3 corresponding to the 3 layers of colors).

trainData = np.true_divide(trainData, 255.0).reshape(len(trainData), 32, 32, 3)
testData = np.true_divide(testData, 255.0).reshape(len(testData), 32, 32, 3)

# Faced issue with the TF model not accepting the labels as lists and thus had to convert them to numpy arrays.
trainLabels = np.array(trainLabels)
testLabels = np.array(testLabels)


# NEURAL NETWORK ARCHITECTURE


model = models.Sequential()

# 4 sets of VGG Blocks to allow for feature extraction, each consisting of:
#     1 Convolutional Layer
#     1 MaxPooling Layer

model.add(layers.Conv2D(64, (3,3), activation = "relu", input_shape = (32,32,3), padding = "same"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation = "relu", padding = "same"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation = "relu", padding = "same"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation = "relu", padding = "same"))
model.add(layers.MaxPooling2D(2,2))


model.add(layers.Flatten())
# We use Dropout to prevent any possible overfitting
model.add(layers.Dropout(0.3))
model.add(layers.Dense(512, activation = "relu"))


# THE FINAL LAYER:
#     10 Neurons, corresponding to the 10 classes of the CIFAR-10 Dataset
#     Using the softmax activation function to clearly output probability of the image being from either one of the 10 classes.
model.add(layers.Dense(10, activation = "softmax"))


model.compile(optimizer="adam", loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])

# Training the NN using the CIFAR-10 Dataset, with information on accuracy across each epoch being stored in 'epochHistory'
epochHistory = model.fit(trainData, trainLabels, epochs = 6, validation_data = (testData,testLabels))

# Finally Plotting a graph of the loss for both the training and validation sets
pyplot.plot(epochHistory.history['loss'], label = "Training Loss")
pyplot.plot(epochHistory.history['val_loss'], label = "Validation Loss")
pyplot.xlabel("Epoch")
pyplot.ylabel("Loss")
pyplot.legend(loc = "lower center")
pyplot.show()