"""
This is a starter file to get you going. You may also include other files if you feel it's necessary.

Make sure to follow the code convention described here:
https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions

Hints:
* The internet is your friend! Don't be afraid to search for tutorials/intros/etc.
* We suggest using a convolutional neural network.
* TensorFlow Keras has the CIFAR-10 dataset as a module, so you don't need to manually download and unpack it.
"""

# Import libraries/modules needed
import matplotlib.pyplot as plt
import cv2
%matplotlib inline
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.metrics import classification_report

# loading the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# normalizing pixel values from 0-225 to 0-1
x_train = x_train/225
x_test = x_test/255

# encoding the catagorical data
y_cat_train = to_categorical(y_train,10)
y_cat_test = to_categorical(y_test,10)

#define the model
model = Sequential()

## FIRST SET OF LAYERS

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

## SECOND SET OF LAYERS

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES BEFORE FINAL LAYER
model.add(Flatten())

# 256 NEURONS IN DENSE HIDDEN LAYER
model.add(Dense(256, activation='relu'))

# CLASSIFIER
model.add(Dense(10, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()
model.fit(x_train,y_cat_train,verbose=1,epochs=10)
model.evaluate(x_test,y_cat_test)


predictions = model.predict_classes(x_test)
print(classification_report(y_test,predictions))
                 
