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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
# Your working code here

# Constant Variables
BATCH_SIZE = 64
EPOCHS = 10


def get_data():
    """
    Loads the cifar10 dataset and preprocesses it

    :return: the preprocessed training and testing data
    """
    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

    # Normalize data
    xTrain = tf.keras.utils.normalize(xTrain, axis=1)
    xTest = tf.keras.utils.normalize(xTest, axis=1)

    # Categorize data
    yTrain = tf.keras.utils.to_categorical(yTrain)
    yTest = tf.keras.utils.to_categorical(yTest)
    return xTrain, yTrain, xTest, yTest


def get_model():
    """
    Creates a sequential model with 3 layers and compiles it

    :return: built and compiled model
    """
    # Input Layer
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second Layer
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third Layer
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))

    # Output Layer
    model.add(Dense(10))
    model.add(Activation("softmax"))

    # Compile Model
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model


def plot_loss(history):
    """
    Plots the model's training loss.
    """
    loss = history.history["loss"]

    # Number of epochs is found by taking the length of the acc array
    epochs = range(1, len(loss) + 1)

    plt.clf()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.title("Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_accuracy(history):
    """
    Plots the model's training accuracy.
    """
    acc = history.history["accuracy"]

    # Number of epochs is found by taking the length of the acc array
    epochs = range(1, len(acc) + 1)

    plt.clf()
    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.title("Training accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def main():
    """
    Main method for fitting and testing the model accuracy

    :return: None
    """

    # Get data and model
    xTrain, yTrain, xTest, yTest = get_data()
    model = get_model()

    # Fit the data
    history = model.fit(xTrain, yTrain, batch_size=BATCH_SIZE, epochs=EPOCHS)

    # Plot data
    plot_accuracy(history)
    plot_loss(history)

    # Evaluate model
    testLoss, testAccuracy = model.evaluate(xTest, yTest)

    # Print the loss and accuracy
    print(f"Loss: {testLoss}")
    print(f"Accuracy: {testAccuracy * 100}%")


if __name__ == "__main__":
    main()
