"""
This is a starter file to get you going. You may also include other files if you feel it's necessary.

Make sure to follow the code convention described here:
https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions

Hints:
* The internet is your friend! Don't be afraid to search for tutorials/intros/etc.
* We suggest using a convolutional neural network.
* TensorFlow Keras has the CIFAR-10 dataset as a module, so you don't need to manually download and unpack it.
"""

import matplotlib.pyplot as plt
import cv2
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.metrics import classification_report

def load_data():
    # loading the dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # normalizing pixel values from 0-225 to 0-1
    x_train = x_train/225
    x_test = x_test/255

    # encoding the catagorical data
    y_cat_train = to_categorical(y_train,10)
    y_cat_test = to_categorical(y_test,10)

    return x_train, x_test, y_cat_train, y_cat_test

def def_model():
    #define the model
    model = Sequential()

    ## FIRST SET OF LAYERS

    # CONVOLUTIONAL LAYER
    model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
    # CONVOLUTIONAL LAYER
    model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))

    # POOLING LAYER
    model.add(MaxPool2D(pool_size=(2, 2)))

    ## SECOND SET OF LAYERS

    # CONVOLUTIONAL LAYER
    model.add(Conv2D(filters=64, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
    # CONVOLUTIONAL LAYER
    model.add(Conv2D(filters=64, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))

    # POOLING LAYER
    model.add(MaxPool2D(pool_size=(2, 2)))

    # FLATTEN IMAGES BEFORE FINAL LAYER
    model.add(Flatten())

    # 512 NEURONS IN DENSE HIDDEN LAYER 
    model.add(Dense(512, activation='relu'))

    # LAST LAYER, CLASSIFIER
    model.add(Dense(10, activation='softmax'))


    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])



    model.summary()
    return model
    
    
# load dataset
x_train, y_cat_train, x_test, y_cat_test = load_dataset()
# define model
model = def_model()
# fit model
model.fit(x_train,y_cat_train,verbose=1,epochs=1)
#evaluate
model.evaluate(x_test,y_cat_test)
# predict
predictions = model.predict_classes(x_test)
print(classification_report(y_test,predictions))

                 

Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 29, 29, 32)        1568      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 26, 26, 32)        16416     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 10, 10, 64)        32832     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 7, 7, 64)          65600     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 3, 3, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 576)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               295424    
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5130      
=================================================================
Total params: 416,970
Trainable params: 416,970
Non-trainable params: 0
_________________________________________________________________
Epoch 1/20
50000/50000 [==============================] - 238s 5ms/step - loss: 1.5903 - acc: 0.4251
Epoch 2/20
50000/50000 [==============================] - 234s 5ms/step - loss: 1.1703 - acc: 0.5931
Epoch 3/20
50000/50000 [==============================] - 238s 5ms/step - loss: 1.0092 - acc: 0.6535
Epoch 4/20
50000/50000 [==============================] - 237s 5ms/step - loss: 0.9432 - acc: 0.6792
Epoch 5/20
50000/50000 [==============================] - 241s 5ms/step - loss: 0.9195 - acc: 0.6914
Epoch 6/20
50000/50000 [==============================] - 244s 5ms/step - loss: 0.8986 - acc: 0.6984
Epoch 7/20
50000/50000 [==============================] - 230s 5ms/step - loss: 0.8904 - acc: 0.7025
Epoch 8/20
50000/50000 [==============================] - 250s 5ms/step - loss: 0.8939 - acc: 0.7040
Epoch 9/20
50000/50000 [==============================] - 273s 5ms/step - loss: 0.8814 - acc: 0.7082
Epoch 10/20
50000/50000 [==============================] - 254s 5ms/step - loss: 0.8761 - acc: 0.7105
Epoch 11/20
50000/50000 [==============================] - 249s 5ms/step - loss: 0.8697 - acc: 0.7135
Epoch 12/20
50000/50000 [==============================] - 255s 5ms/step - loss: 0.8639 - acc: 0.7177
Epoch 13/20
50000/50000 [==============================] - 255s 5ms/step - loss: 0.8599 - acc: 0.7190
Epoch 14/20
50000/50000 [==============================] - 267s 5ms/step - loss: 0.8565 - acc: 0.7210
Epoch 15/20
50000/50000 [==============================] - 252s 5ms/step - loss: 0.8537 - acc: 0.7217
Epoch 16/20
50000/50000 [==============================] - 263s 5ms/step - loss: 0.8509 - acc: 0.7250
Epoch 17/20
50000/50000 [==============================] - 258s 5ms/step - loss: 0.8299 - acc: 0.7322
Epoch 18/20
50000/50000 [==============================] - 276s 6ms/step - loss: 0.8291 - acc: 0.7312
Epoch 19/20
50000/50000 [==============================] - 254s 5ms/step - loss: 0.8363 - acc: 0.7324
Epoch 20/20
50000/50000 [==============================] - 248s 5ms/step - loss: 0.8136 - acc: 0.7369
10000/10000 [==============================] - 14s 1ms/step
             precision    recall  f1-score   support

          0       0.72      0.65      0.68      1000
          1       0.81      0.83      0.82      1000
          2       0.36      0.71      0.48      1000
          3       0.65      0.22      0.33      1000
          4       0.45      0.71      0.56      1000
          5       0.87      0.30      0.44      1000
          6       0.57      0.85      0.69      1000
          7       0.93      0.41      0.57      1000
          8       0.72      0.83      0.77      1000
          9       0.88      0.66      0.75      1000

avg / total       0.70      0.62      0.61     10000
