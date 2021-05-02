
"""
  File: model.py
  Author: UWARG
 
  Section: LICENSE
 
  Copyright (c) 2021-2022, Waterloo Aerial Robotics Group (WARG)
  All rights reserved.
 
  This software is licensed under a modified version of the BSD 3 clause license
  that should have been included with this software in a file called COPYING.txt
  Otherwise it is available at:
  https://raw.githubusercontent.com/UWARG/computer-vision/master/COPYING.txt

 """

#Importing libraries
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from keras.datasets import cifar10
from keras.callbacks import ReduceLROnPlateau 
from keras.losses import SparseCategoricalCrossentropy
import numpy as np 
import matplotlib.pyplot as plt 

# Globlal variables that can easily be changed to the requirements of the program
IMG_HEIGHT = 32
IMG_WIDTH = 32
EPOCHS = 25
BS = 20 #Batch size => refers to the number of training examples utilized in one iteration

#Load cifar10 dataset into training and testing images and labels to be used for the CNN
(trainImages, trainLabels), (testImages, testLabels) = cifar10.load_data()

#Shrink pixel values to binary (from a value between 0 to 255 to 0 to 1)
trainImages = trainImages / 255.0
testImages = testImages / 255.0

#Class labels for the dataset (for reference)
classNames = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Now onto the funstuff - The construction of the CNN using 32 nodes and gradually increase the CNN's neurons (in the hidden layer) to increase the quality of the CNN
# and of course it is used with the max pool function witha  kernal size of 2x2 to pool the convolutions.

"""
Note: The convolution and pooling steps can be repeated multiple times to extract additional features and reduce the size of the input to 
the neural network. One of the benefits of these processes is that, by convoluting and pooling, the neural network becomes less sensitive to variation. 
That is, if the same picture is taken from slightly different angles, the input for convolutional neural network will be similar, whereas, without 
convolution and pooling, the input from each image would be vastly different.

"""
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))) 
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2))) 
model.add(layers.Conv2D(32, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2))) 
model.add(layers.Conv2D(128, (3, 3), activation='relu'))  
model.add(layers.Conv2D(128, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2))) 
model.add(layers.Flatten()) 
model.add(layers.Dense(64, activation="softmax")) # softmax was used instead of relu as it is ideally suited for multiclass problems (such as this one!)
model.add(layers.Dropout(0.7))  # Droput is done so that the model doesnt overfit with the data.
model.add(layers.Dense(10)) 
# model.summary() # You can include this line if you want to see the models' network and parameters 

#Configure the model using adam optimizer, along with Sparse Categorical Cross entropy and and accuracy metric 

model.compile(optimizer = 'adam', loss = SparseCategoricalCrossentropy(from_logits = True), metrics = ["accuracy"])

# Trains the model for a fixed number of epochs with 20 epochs and a batch size of 32. After training, the model goes to compare with
# a validation set which is the same as the set of testing images and labels we had made earlier
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
history = model.fit(trainImages, trainLabels, epochs= EPOCHS, validation_data = (testImages, testLabels), callbacks=[reduceLR])

# Saving model to disk (can be commented off to see the model)
# model.save('CNN_CIFAR.h5') # Remove the first hash to save the model if you would like

# Plot code used to create line graph showing accuracy vs val accuracy
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

"""
References for software integrity (not really required as they are opensource tools, putting it just in case)

Keras Docs - https://keras.io/
Tensorflow Docs - https://www.tensorflow.org/api_docs/python/tf
UWARG Docs
Several stackexchange threads

"""



