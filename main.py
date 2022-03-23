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

import sys
import os

# To remove warning messages in terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from keras.layers import Input, Dense, Flatten, Activation
from keras import layers

def cifar_load():
    """
    Returns training and test data from CIFAR-10 dataset

    Parameters
    ----------
    None

    Returns
    -------
    (x_train, y_train), (x_test, y_test): Tuple of NumPy arrays
        uint8 NumPy arrays representing grayscale image data (x) and labels (integers 1 - 9) (y)
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # to reduce the complexity of computations in neural network
    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train), (x_test, y_test)

def seq_model():
    """
    Returns convolutional neural network with a sequential model

    Parameters
    ----------
    None

    Returns
    -------
    model: keras.Sequential
        Convolutional neural network with a sequential model
    """
    model = keras.Sequential()

    # Instantiate 32 x 32 images as a Keras tensor
    model.add(keras.Input(shape=(32, 32, 3)))

    # Add convolutional layers, 32 -> 64 filters each, with relu activation for a simple linear NN
    # Block 1 of Layers
    model.add(layers.Conv2D(32, kernel_size=(3,3), strides=(2, 2), activation='relu'))
    model.add(layers.MaxPooling2D(2))

    # Block 2 of Layers
    model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2))
    
    # Flatten tensor shape to one-dimension with average pooling to avoid overfitting
    model.add(layers.GlobalAveragePooling2D())
    
    # Process data using Dense operator
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile model with adam algorithm due to large dataset, and a cross-entropy loss function
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def plot_model(history):
    """
    Plots validation, accuracy, and loss data of the CNN over epochs

    Parameters
    ----------

    History object returned by the fit method applied to the CNN

    Returns
    -------
    None
    """
    
    # Summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def cnn_main():
    """
    Calls data loading, sequential model, and plotting functions

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Call data loading function
    (x_train, y_train), (x_test, y_test) = cifar_load()

    # Call sequential model function to retrieve model
    model = seq_model()
    
    # Fit the model to create history object
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), validation_split=0.33, epochs=50, batch_size=10, verbose=1)
    
    # Call plotting function with history object
    plot_model(history)

# Call main function
cnn_main()


