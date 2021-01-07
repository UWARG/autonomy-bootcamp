#import all main and helper libraries
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import numpy as np 
import matplotlib.pyplot as plt 

#get cifar-10-dataset
(trainImages, trainLabels), (testImages, testLabels) = datasets.cifar10.load_data()

#Shrink pixel values from 0-255 to 0-1 for ease of translation onto relu 
trainImages = trainImages / 255.0
testImages = testImages / 255.0

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

#Note: all convolving functions start with input shape of 32, 32, 3 for 32x32 px input and RGB colouring
#Note2: relu activation function used 
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) #Layer 1: convolutional base translates 3x3 sections onto single cell; changes 32x32 into 30x30
model.add(layers.MaxPooling2D((2, 2))) #Layer 2: more rapid condensing; divides matrix from last layer into 2x2 sections and one-to-one projects onto 15x15 matrix
model.add(layers.Conv2D(64, (3, 3), activation='relu')) #Layer 3: repeat convolutional base; 15x15 to 13x13 by translating 3x3 sections
model.add(layers.MaxPooling2D((2, 2))) #Layer 4: rapid condensing pt.2; divides 13x13 matrix int 6x6 (2x2 sections overlap near end to compensate for odd-dimension input)
model.add(layers.Conv2D(64, (3, 3), activation='relu')) #Layer 5: last convolutional iteration: 6x6 translated into more manageable 4x4 
model.add(layers.Flatten()) #flattens 64 channels of 4x4 array input into singular 1024 straight nodes
model.add(layers.Dense(64, activation="relu")) #fully connected layer of 64 nodes
model.add(layers.Dense(10)) #output layer of 10 for aforementioned 10 classes
model.summary()

#Adam optimizer used; loss function translates 'closeness' of model to changes in bias and weight
#accuracy used as an indicator of the model's efficiency
model.compile(optimizer = "adam", loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ["accuracy"])

#5 epochs used; reasonable amount to balance cpu usage and diminishing returns
history = model.fit(trainImages, trainLabels, epochs = 5, validation_data = (testImages, testLabels))
#epoch = no. of times same image shows; but different order

#matplot used to create line graph showing accuracy
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1]) #accuracy axis chosen to be between 0.5 and 1 
plt.legend(loc='lower right')
plt.show()

#testLoss, testAcc = model.evaluate(testImages,  testLabels, verbose=2)
#print(testAcc)

#By: Prabhav Desai