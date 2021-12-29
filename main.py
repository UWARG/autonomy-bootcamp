"""
An image classifier for the CIFAR-10 dataset created using Python and
Tensorflow. Submission for UWARG's computer vision bootcamp.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# The images belong to one of the following classes: airplane,
# automobile, bird, cat, deer, dog, frog, horse, ship or truck
NUM_CLASSES = 10

# Tunable batch size for training
BATCH_SIZE = 64

# Using a relatively high-number of epochs so that overfitting occurs
# and can be addressed with the EarlyStopping callback
EPOCHS = 50

# There are 50000 training images in the dataset, so using a
# validation split of 0.2 uses 10000 images for validation, which is
# about 17% of the total dataset
VALIDATION_SPLIT = 0.2


def load_data():
    """
    Loads the CIFAR-10 dataset and preprocesses it.

    :return: Preprocessed training and test data
    """
    # Load CIFAR-10 dataset from Keras
    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

    # Normalize images by scaling values to be in the range [0, 1]
    xTrain = xTrain.astype("float32") / 255
    xTest = xTest.astype("float32") / 255

    # Perform one-hot encoding since this is a multi-class
    # classification problem
    yTrain = to_categorical(yTrain)
    yTest = to_categorical(yTest)

    return (xTrain, yTrain), (xTest, yTest)


def build_model():
    """
    Build and compile the model. The model architecture is a simple
    CNN structure, consisting of 3 convolution layers and 3 max
    pooling layers.

    :return: Built and compiled model
    """
    model = models.Sequential([
        # Conv layers are added with increasing filter counts
        layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
        layers.MaxPool2D(pool_size=2),
        layers.Dropout(0.35),
        layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
        layers.MaxPool2D(pool_size=2),
        layers.Dropout(0.35),
        layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
        layers.Dropout(0.35),
        layers.Flatten(),
        # Softmax activation for the last layer so that a probability
        # distribution vector is created for each possible class
        layers.Dense(10, activation="softmax")
    ])

    # We use the categorical_crossentropy loss function since the
    # target data is one-hot encoded
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model


def plot_loss(history):
    """
    Plots the model's training and validation loss.
    """
    loss = history.history["loss"]
    valLoss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)

    plt.clf()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, valLoss, "b", label="Validation loss")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_accuracy(history):
    """
    Plots the model's training and validation accuracy.
    """
    acc = history.history["accuracy"]
    valAcc = history.history["val_accuracy"]
    epochs = range(1, len(acc) + 1)

    plt.clf()
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, valAcc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def main():
    # Get the preprocessed CIFAR-10 dataset
    (xTrain, yTrain), (xTest, yTest) = load_data()

    # Get a compiled model
    model = build_model()

    # EarlyStopping callback use to account for overfitting
    earlyStop = callbacks.EarlyStopping(monitor="val_loss", patience=2)

    # Train model
    history = model.fit(
        xTrain,
        yTrain,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=[earlyStop]
    )

    # Plot results
    plot_loss(history)
    plot_accuracy(history)

    # Evaluate model on test data
    print()
    testLoss, testAcc = model.evaluate(xTest, yTest)

    # Print results
    print()
    print(f"{testLoss=}")
    print(f"{testAcc=}")


if __name__ == "__main__":
    main()
