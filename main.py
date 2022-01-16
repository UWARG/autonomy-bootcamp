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
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models
from keras.layers import Dropout, BatchNormalization



def data_preprocessing(train_data, test_data):
    # Normalize pixel values to be between 0 and 1
    # This is so that it trains faster (or so I have read)
    train_data, test_data = train_data / 255.0, test_data / 255.0

def build__and_compile_model(model):
    # Input shape is due to CIFAR10 dataset input, activation function being relu trains faster 
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    # Pooling layer is added to "consolidate" learning done by convolutional layer
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # I try to reduce overfitting using dropout (I actually don't know if this is even necessary).
    model.add(Dropout(0.2))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # Flatten the 3D output into 1D so that dense layers can take them as input
    model.add(layers.Flatten())
    # Apparently, adding more dense layers improves classification. Mathematically, I don't yet know why.
    model.add(layers.Dense(64, activation='relu'))
    # Create a final dense layer with 10 outputs, since CIFAR has 10 output classes.
    model.add(layers.Dense(10))
    # According to the tensorflow website, this is how to compile a neural network.
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

def train_model(model, train_data, train_labels, val_data, val_labels):
    # Training using training dataset, validating using validation dataset, over 
    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))
    return(history)


def plot_model_history(history):
    # Training and validation loss over epochs
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def print_model_accuracy(model):
    # get testing accuracy
    testAcc = model.evaluate(test_images,  test_labels, verbose=2)
    # print testing accuracy
    print(testAcc)



# Get the data from CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# preprocess data
data_preprocessing(train_images, test_images)

# I used a sequential model because I do not know how to use non-sequential models
model = models.Sequential()

# build, compile, and train model
build__and_compile_model(model)
history = train_model(model, train_images, train_labels, test_images, test_labels)

# plot model training history, and print accuracy
plot_model_history(history)
print_model_accuracy(model)

