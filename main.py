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
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import (
    Dense, 
    Dropout, 
    MaxPooling2D, 
    BatchNormalization, 
    Input, 
    Conv2D, 
    Flatten
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
import argparse

# Your working code here
# we are going to keep this as a global for this file
CIFAR_DICT = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

# 1. loading the data
def load_data():
    """
    Loads the Cifar10 Dataset from Tensorflow's pre-loaded datasets
    
    Parameters
    ----------
    None

    Returns
    -------
    xTrain : np.array, shape = (50000, 32, 32, 3)
        Returns a numpy array that holds the x training data 
    xTest : np.array, shape = (10000, 32, 32, 3)
        Returns a numpy array that holds the x testing data 
    yTrain : np.array, shape = (50000, 1)
        Returns a numpy array that holds the y training data 
    xTrain : np.array, shape = (10000, 1)
        Returns a numpy array that holds the y testing data 
    """

    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

    xTrain = np.array(xTrain)
    xTest = np.array(xTest)
    yTrain = np.array(yTrain)
    yTest = np.array(yTest)

    return xTrain, xTest, yTrain, yTest


# 2. understanding the images
def visualize_images(x, y, name="visualize"):
    """
    Function that takes 9 images and labels and plots them using matplotlib to understand the data
    
    Parameters
    ----------
    x : np.array
        A numpy array containing a minimum of 9 images corresponding to the data
    y : np.array
        A numpy array containing a minimum of 9 labels corresponding to the data and 
        matching with the x array passed into the function
    name : string
        A string to determine the filename that the visualization will be saved in

    Returns
    -------
    None
    """

    f, axarr = plt.subplots(3,3, figsize=(12,12))
    for i in range(3):
        for j in range(3):
            axarr[i,j].imshow(x[i+j])
            axarr[i,j].title.set_text(CIFAR_DICT[y[i+j][0]])
    f.tight_layout()
    plt.savefig(f"images/{name}_images.png")
    plt.clf()
    plt.cla()


# 3. creating a convolutional model
def cifar_conv_model():
    """
    Returns a not compiled convolutional model specifically made for the cifar10 dataset
    
    Parameters
    ----------
    None

    Returns
    -------
    model : tensorflow.keras.models.Model
        A not compiled model defined within the function specifying the input and outputs
    """

    inp =  Input((32, 32, 3))
    x = BatchNormalization()(inp)
    
    x = Conv2D(64, kernel_size=3, strides=(1,1), activation='relu', kernel_initializer='glorot_normal')(x)
    x = Conv2D(64, kernel_size=3, strides=(1,1), activation='relu', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D(pool_size=(3,3))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, kernel_size=2, strides=(1,1), activation='relu', kernel_initializer='glorot_normal')(x)
    x = Conv2D(256, kernel_size=2, strides=(1,1), activation='relu', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(10, activation='softmax')(x)
    
    model = Model(inp, out)

    print(model.summary())
    return model


# 4. returning the compiled the model
def compiled_model(opt=None, loss=None, metrics=None):
    """
    Returns a compiled tensorflow.keras model
    
    Parameters
    ----------
    opt : tensorflow.keras.optimizers
        default = None
        The optimizer that is used to compile the model
    loss : tensorflow.keras.losses
        default=None
        The loss metric to be used for compiling the model
    metrics : list<tensorflow.keras.metrics>
        default = None
        The metrics to be used within the model

    Returns
    -------
    model : tensorflow.keras.models.Model
        A compiled model defined within the function specifying the input and outputs
    """

    model = cifar_conv_model()

    if opt is None:
        opt = Adam(learning_rate=0.001)
    if loss is None:
        loss = SparseCategoricalCrossentropy()
    if metrics is None:
        metrics = ['accuracy']

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
    )

    return model


# 5. producing callbacks
def standard_callbacks():
    """
    Returns a list of two standard callbacks
    
    Parameters
    ----------
    None

    Returns
    -------
    list<tensorflow.keras.callbacks>
        A list of callbacks to be used when fitting the model
    """

    lrReduction = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.1, 
        patience=4, 
        verbose=1,
        mode='auto',
        min_delta=0.0001, 
        cooldown=0, 
        min_lr=0.0000001
    )
    checkpoint = ModelCheckpoint(
        "model/cifar10_main_model.h5", 
        save_best_only=True,
        verbose=1
    )

    callbacks = [
        lrReduction,
        checkpoint
    ]

    return callbacks


# 6. plotting functions
def acc_plot(history):
    """
    Plots the accuracy of the model
    
    Parameters
    ----------
    history : tensorflow.keras.callbacks
        The output produced when fitting the model

    Returns
    -------
    None
    """

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("images/model_acc.png")
    plt.clf()
    plt.cla()


def loss_plot(history):
    """
    Plots the loss of the model
    
    Parameters
    ----------
    history : tensorflow.keras.callbacks
        The output produced when fitting the model

    Returns
    -------
    None
    """

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("images/model_loss.png", bbox_inches='tight')
    plt.clf()
    plt.cla()


if __name__ == '__main__':
    # using an argument parser to give the user functionality
    # to run the function
    parser = argparse.ArgumentParser(
        description='Process the inputs'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        help='how many epochs',
        default=20
    )
    parser.add_argument(
        '--verbose', 
        type=int, 
        help='0,1,2',
        default=1
    )

    args = parser.parse_args()
    epochs_ = args.epochs
    verbose_ = args.verbose

    # loading in the data as np arrays
    xTrain, xTest, yTrain, yTest = load_data()

    # understanding the images
    visualize_images(xTrain, yTrain, "train")
    visualize_images(xTest, yTest, "test")

    # loading in the compiled model
    # summary is printed and we use Adam opt, SparseCC loss
    # and standard acc metrics
    model = compiled_model()

    # generating callbacks
    callbacks = standard_callbacks()

    # fitting the model
    history = model.fit(
        xTrain, 
        yTrain,
        validation_data=(xTest, yTest),
        epochs=epochs_,
        verbose=verbose_,
        batch_size=32,
        callbacks=callbacks
    )

    # producing the accuracy and the loss plots
    acc_plot(history)
    loss_plot(history)
