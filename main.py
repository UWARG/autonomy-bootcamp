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
import pickle

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10

# Your working code here
def prep(x, y):
    #normalize the data to a [0,1] interval using min-max normalizatoin
    x = (x-np.min(x))/(np.max(x) - np.min(x))
    #print(x)

    #one hot encoding to represent each category as numerical values
    from tensorflow.keras.utils import to_categorical
    y = to_categorical(y)
    #print(y)
    return x, y

#defining the CNN model
#sequential is used for layers with a single input and is created by layers
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot(history):
    #plot loss over epochs
    epochs = range(1, len(history.history['loss']) + 1)
    plt.plot(epochs, history.history['loss'], color='blue', label='train')
    plt.plot(epochs, history.history['val_loss'], color='green', label='test')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()
    
    print()
    #plot accuracy over epochs
    epochs = range(1, len(history.history['loss']) + 1)
    plt.plot(epochs, history.history['accuracy'], color='blue', label='train')
    plt.plot(epochs, history.history['val_accuracy'], color='green', label='test')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()

def run():
    #load in the data batches
    #x, y, xTest, yTest = load_batches()
    (x, y), (xTest, yTest) = cifar10.load_data()

    #printing first 9 images in the dataset
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x[i])
    plt.show()
    
    #prep the data for training by standardizing and one hot encoding
    x, y = prep(x,y)
    xTest, yTest = prep(xTest, yTest)
    
    #define the model for the neural network
    model = define_model()
    
    #print the model's summary
    model.summary()
    
    #training the model
    history = model.fit(x, y, epochs=10, batch_size=64, validation_data=(xTest, yTest), verbose=1)
    
    print()    
    #printing loss over epochs and accuracy over epochs
    plot(history)

run()
