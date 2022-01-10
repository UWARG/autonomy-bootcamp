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
                 

  170500096/170498071 [==============================] - 28s 0us/step
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 29, 29, 32)        1568      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 11, 11, 32)        16416     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 32)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 800)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               205056    
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 225,610
Trainable params: 225,610
Non-trainable params: 0
_________________________________________________________________
Epoch 1/10
50000/50000 [==============================] - 66s 1ms/step - loss: 1.5302 - acc: 0.4508
Epoch 2/10
50000/50000 [==============================] - 63s 1ms/step - loss: 1.1867 - acc: 0.5841
Epoch 3/10
50000/50000 [==============================] - 64s 1ms/step - loss: 1.0395 - acc: 0.6386
Epoch 4/10
50000/50000 [==============================] - 67s 1ms/step - loss: 0.9363 - acc: 0.6749
Epoch 5/10
50000/50000 [==============================] - 62s 1ms/step - loss: 0.8511 - acc: 0.7060
Epoch 6/10
50000/50000 [==============================] - 67s 1ms/step - loss: 0.7952 - acc: 0.7261
Epoch 7/10
50000/50000 [==============================] - 66s 1ms/step - loss: 0.7431 - acc: 0.7450
Epoch 8/10
50000/50000 [==============================] - 63s 1ms/step - loss: 0.7026 - acc: 0.7612
Epoch 9/10
50000/50000 [==============================] - 65s 1ms/step - loss: 0.6655 - acc: 0.7742
Epoch 10/10
50000/50000 [==============================] - 64s 1ms/step - loss: 0.6352 - acc: 0.7863
10000/10000 [==============================] - 5s 462us/step
  
  
               precision    recall  f1-score   support

          0       0.67      0.70      0.69      1000
          1       0.85      0.77      0.81      1000
          2       0.51      0.60      0.55      1000
          3       0.45      0.53      0.48      1000
          4       0.67      0.57      0.61      1000
          5       0.60      0.47      0.53      1000
          6       0.71      0.77      0.74      1000
          7       0.73      0.72      0.72      1000
          8       0.72      0.79      0.75      1000
          9       0.81      0.70      0.75      1000

avg / total       0.67      0.66      0.66     10000
