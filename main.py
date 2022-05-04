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
import logging

import numpy as np
from keras.datasets.cifar10 import load_data
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

# Your working code here
def load_and_preprocess_data():
    """
    Returns CIFAR10 dataset with labels as one-hot vectors and pixel data is scaled down by 255.

    Returns
    --------
    xTrain : np.ndarray
        training data for the model
    yTrain : np.ndarray
        training labels for the model
    xTest : np.ndarray
        testing data for the model
    yTest : np.ndarray
        testing labels for the model
    """
    # Importing the data
    (xTrain, yTrain), (xTest, yTest) = load_data()

    # Converting labels into one hot encoders
    yTrain = to_categorical(yTrain, dtype='uint8')
    yTest = to_categorical(yTest, dtype='uint8')

    # Scaling down the input data by a factor of 255
    xTrain = xTrain.astype('float32')
    xTest = xTest.astype('float32')
    xTrain = xTrain / 255.0
    xTest = xTest / 255.0
    return xTrain, yTrain, xTest, yTest


def build_model():
    """
    Builds a Sequential model with custom layer.

    Return
    --------
    model : Sequential
        model used for cifar10 image classification
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    logging.log(20, 'Model built successfully')
    return model


def create_plots(history):
    """
    Creates plots to track loss and accuracy over epochs

    Parameters
    -----------
    history : callbacks.History

    """
    # Plot for the loss
    fig = plt.figure(figsize=(10, 6))
    plt.title('Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.legend()
    plt.savefig('C:/Users/Zeyad/Desktop/Portfolio/loss.png', format='png', dpi=200)
    plt.close(fig)

    # Plot for the accuracy
    fig = plt.figure(figsize=(10, 6))
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.legend()
    plt.savefig('C:/Users/Zeyad/Desktop/Portfolio/accuracy.png', format='png', dpi=200)
    plt.close(fig)


xTrain, yTrain, xTest, yTest = load_and_preprocess_data()
model = build_model()
history = model.fit(xTrain, yTrain, epochs=50, batch_size=128, validation_data=(xTest, yTest), verbose=1)
create_plots(history)
acc = model.evaluate(xTest, yTest)
print("The accuracy of the model on testing data is {}".format(round(acc[1], 3)*100))