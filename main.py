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
import matplotlib.pyplot as plt
from keras.datasets import cifar10
import tensorflow as tf


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.constraints import maxnorm


# Your working code here


"""
Function: data_load()

Loads data from the CIFAR10 dataset (using the TF Keras API) and obtains the training and testing datasets

Parameters:
None

Return:
Training and Testing data (X and Y) [tuple]

"""


def data_load():
    (trainingX, trainingY), (testX, testY) = cifar10.load_data()

    return trainingX, trainingY, testX, testY


"""
Function: one_hot_encode()

As we have only ten different classes, we can make an identifcation system for each class by using One Hot Encoding for the target values (trainingY and testY)
One Hot Encoding allows us to map each class with a unique 10 digit (as we have 10 classes) number
Eg: Airplane = 0000000001. Keras has a utility function called to_categorical() that does this for you

Parameters:
Training and Testing tuples to be encoded

Return:
Tuples of encoded target training and testing data, with the shape of test [classes]

"""


def one_hot_encode(training, test):
    training = tf.keras.utils.to_categorical(training)
    test = tf.keras.utils.to_categorical(test)
    classes = test.shape[1]

    return training, test, classes


"""

Function: rescale_pixels()

Each image in the CIFAR10 dataset have pixel values ranging from 0 to 255
To help us scale the actual model, it will be beneficial to put it in a range of [0,1]. To achieve this, we first convert the data into floats and then divide by 255.

Parameters:
Training and Test tuples to be rescaled

Return:
Tuples of rescaled Training and Testing data

"""


def rescale_pixels(training, test):
    # the data has to be converted from integers to floats
    training_rs = training.astype('float32')
    test_rs = test.astype('float32')

    # rescaling by dividing by 255
    training_rs = training_rs/255
    test_rs = test_rs/255

    shape = training_rs.shape[1:]
    return training_rs, test_rs, shape


"""
Function: create_model()

We create a Sequential Keras model so that we can build the model layer by layer.
We create blocks by using multiple 3x3 filters which is followed by a max-pooling layer; then we stack up these blocks to form our baseline model.
The model also has dropout layers to avoid overfitting and a flattening layer to convert the feature map to one-dimension

Parameters:
Shape of trainingX [trainingX.shape[1:]] and Classes of testY [testY.shape[1]]

Returns:
The keras model
"""


def create_model(shape, classes):
    # sequential model
    model = Sequential()

    # First Block
    # Convolutional layer with 32 3x3 filters and ReLU activation function
    model.add(Conv2D(32, (3, 3), input_shape=shape,
              activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), input_shape=(32, 3, 3),
              activation='relu', padding='same'))
    # Dropout layer to prevent overfitting
    model.add(Dropout(0.2))
    # Batch Normalization layer
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), input_shape=(32, 3, 3),
              activation='relu', padding='same'))
    # max-pooling layer to reduce computational complexity of the network
    model.add(MaxPooling2D((2, 2), padding='same'))

    # Second block, increasing the depth to 64
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    # Third block, increasing the depth to 128
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    # Flattening the model
    model.add(Flatten())

    # Dropout layer to prevent overfitting
    model.add(Dropout(0.2))
    # Dense layer to intialize a connected network
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))

    # More dropout and dense layer
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    # Softmax Dense layer which uses the classes obtained from testY
    model.add(Dense(classes, activation='softmax'))

    # Compiling model with Categorical cross-entropy, Adam optimizer, and with an accuracy metric
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # retunring the model
    return model


"""
Function: plot_loss()

Plotting a Loss vs Epochs graph using the trained model history

Parameters:
Fitted model history

Returns:
Produces a Loss PNG Image
"""


def plot_loss(history):

    # plotting the loss data from the history
    plt.plot(history.history["loss"], color="r")
    plt.plot(history.history["val_loss"], color="b")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["training", "test"])
    plt.savefig("loss.png")
    plt.close()


"""
Function: plot_accuracy()

Plotting a Accuracy vs Epochs graph using the trained model history

Parameters:
Fitted model history

Returns:
Produces a Accuracy PNG Image
"""


def plot_accuracy(history):

    # plotting the accuracy data from the history
    plt.plot(history.history["accuracy"], color="r")
    plt.plot(history.history["val_accuracy"], color="b")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["training", "test"])
    plt.savefig("accuracy.png")
    plt.close()


"""
Function: run_functions()

Runs all of the functions

Parameters:
None

Returns:
None
"""


def run_functions():

    # loading the data
    trainingX, trainingY, testX, testY = data_load()

    # Pre-procesing the data
    trainingY, testY, classes = one_hot_encode(trainingY, testY)
    trainingX, testX, shape = rescale_pixels(trainingX, testX)

    # Creating the model
    model = create_model(shape, classes)

    # Training the model with 25 Epochs
    history = model.fit(trainingX, trainingY, validation_data=(
        testX, testY), epochs=25, batch_size=32)
    scores = model.evaluate(testX, testY, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    # Plotting the Loss and Accuracy graphs
    plot_loss(history)
    plot_accuracy(history)


run_functions()
