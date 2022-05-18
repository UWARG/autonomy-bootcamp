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

import tensorflow as tf
from tensorflow import keras
from keras import datasets, models, layers
import matplotlib.pyplot as plt


# Your working code here

def load_data():
    """
    This functions loads the data onto training and testing variables

    It also normalizes the xTrain and yTrain variables to be between 0 and 1,
    by dividing them by 255

    return
    --------
    returns a tuple of testing and training values for x and y
    """
    #  loading test and train data
    (xTrain, yTrain),(xTest, yTest) = datasets.cifar10.load_data()

    #  normalizing xTrain and yTrain values to be between 0 and 1
    NORM_CONSTANT = 255
    xTrain = xTrain/NORM_CONSTANT
    xTest = xTest/NORM_CONSTANT

    #  returning training and testing data sets after normalization
    return (xTrain, yTrain),(xTest, yTest)


def build_model(xTrain):
    """
    This function creates a convolutional neural network to
    classify images from the CIFAR-10 dataset

    requires
    --------
    xTrain = training dataset from cifar10
    yTest = test cases answers from cifar10

    return
    ------
    returns the model with all layers setup
    """

    #  creates a sequential model
    classModel = tf.keras.Sequential()

    #  adding the first convolutional layer
    classModel.add(
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=xTrain.shape[1:])
    )

    #  adding first pooling layer
    classModel.add(layers.MaxPooling2D(2, 2))

    #  normalizing the output from the convolution layer
    classModel.add(layers.BatchNormalization())

    #  adding a second convolutional layer
    classModel.add(
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu"
        )
    )

    #  same as above, adding a pooling, and normalization layer
    classModel.add(layers.MaxPooling2D(2, 2))
    classModel.add(layers.BatchNormalization())

    #  flattening input for the hidden layer
    classModel.add(layers.Flatten())

    #  adding the first hidden layer, with 32 neurons
    classModel.add(
        layers.Dense(64, activation="relu")
    )

    #  adding a normalization layer, as before
    classModel.add(layers.BatchNormalization())

    #  adding an output layer
    classModel.add(
        layers.Dense(10, activation="softmax")
    )

    #  compiling the model
    classModel.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return classModel


def plot_data(oldData, dataType):
    """
    This function generates plots of the loss and accuracy of the model

    Requires
    --------
    oldData = previous data from training
    dataType = title of data (accuracy, loss, etc)
    """

    plt.plot(oldData.history[dataType])
    plt.title("model %s" % dataType)
    plt.ylabel(dataType)
    plt.xlabel("epoch#")
    plt.legend(["training results"], loc="upper left")
    plt.show(block=False)
    plt.pause(5)
    plt.close("all")


def main():
    """
    Used to create, train and test cnn, and plot data
    """

    #  loading test data into variables
    (xTrain, yTrain), (xTest, yTest) = load_data()

    #  making model
    classModel = build_model(xTrain)

    #  training the model
    oldData = classModel.fit(
        xTrain, yTrain,
        epochs=10,
        validation_data=(xTest, yTest)
    )
    #  evaluating the model
    classModel.predict(xTest)

    #  plotting the loss data and val_loss data
    plot_data(oldData, "loss")
    plot_data(oldData, "val_loss")


main()
