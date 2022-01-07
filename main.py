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

import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


IMAGE_SIZE = 32
EPOCH_COUNT = 100 #this will take a couple hours, feel free to change for testing
BATCH_SIZE = 64

def load_prep_dataset():
    """
    Loads, normalizes and then splits the CIFAR 10 dataset into train and test
    
    Returns
    --------
    trainXNorm, trainY, testXnorm, testY: np.ndarray
        Returns 4 numpy arrays containing the train and test X and Y data
        from the CIFAR10 dataset
    """
    # load dataset
    (trainX, trainY), (testX, testY) = keras.datasets.cifar10.load_data()
    print("data loaded")
    #normalize data- scale pixel values from 0-255 to 0-1
    trainXFloat= trainX.astype('float32')
    testXFloat = testX.astype('float32')
    trainXNorm = trainXFloat/255.0
    testXNorm = testXFloat/255.0
    return trainXNorm, trainY, testXNorm, testY

def define_model():
    """
    Creates a CNN model for CIFAR image classification
    
    Returns
    --------
    model: tf.keras.Sequential()
        Returns a keras sequential model that can be fit to data from
        load_prep_dataset function
    """
    #create keras model and add CNN layers
    model = models.Sequential()
    model.add(layers.Conv2D(IMAGE_SIZE, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(IMAGE_SIZE*2, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(IMAGE_SIZE*2, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(IMAGE_SIZE*2, activation='relu'))
    #add dense layers, must equal number of output classes
    model.add(layers.Dense(10))
    #compile model with adam optimizer
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']) 
    return model


def plot_loss(history):
    """
    Plots the train and test epoch loss of a keras model

    Params
    --------
    history: keras.callbacks.History, required

    Keras model fit to data
    
    Returns
    --------
    None
    """   
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


model = define_model()

trainX, trainY, testX, testY = load_prep_dataset()

history = model.fit(trainX, trainY, epochs=EPOCH_COUNT, batch_size=BATCH_SIZE, validation_data=(testX, testY))
plot_loss(history)

