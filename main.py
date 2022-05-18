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
import seaborn as sn

# Your working code here

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def add_convolutional_to(model):
    """
    Adds filtering layers to the model variable passed into the function

    Parameters
    ----------
    model : tf.keras.models.Sequential(), required
        Variable size array that represents a neural network
    """
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(
        3, 3), activation=tf.nn.relu, input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))


def add_connected_layer_to(model):
    """
    Adds connected layers to the model variable passed into the function

    Parameters
    ----------
    model : tf.keras.models.Sequential(), required
        Variable size array that represents a neural network
    """
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


def plot_model_data(record, typeOf):
    """
    Plots the model data found in the record variable and specified by the typeOf variable

    Parameters
    ----------
    record : tf.keras.models.Sequential().fit(), required
        Variable size array that represents a neural network
    typeOf : string, required
        Variable that will be used to determine what type of data to plot
    """
    plt.plot(record.history[typeOf])
    plt.title('model %s' % typeOf)
    plt.ylabel(typeOf)
    plt.xlabel('epoch')
    plt.legend(['training progress'], loc='upper left')
    plt.show(block=False)
    plt.pause(5)
    plt.close('all')


def test_model(model):
    """
    Function to run the model variable on the CIFAR-10 test data. Also allows user to check guesses
    made for images in the test data by specifying an index number in the terminal. This function
    ends when a number less than 0 is specified

    Parameters
    ----------
    model : tf.keras.models.Sequential(), required
        Variable size array that represents a neural network
    """
    predictions = model.predict(xTest)
    entered = 0
    while entered > -1:
        entered = int(input())

        # Conditional for when the user decides to exit out of while loop by entering a negative number
        if entered < 0:
            break

        # Printing the guessed class so the user can compare it with the test image
        print(CLASS_NAMES[np.argmax(predictions[entered])])
        plt.imshow(xTest[entered])

        # Using block=False so the image shown will automatically disappear after 2 seconds
        plt.show(block=False)
        plt.pause(2)
        plt.close('all')

    return predictions


# Loading CIFAR-10 data and normalizing the values
cifar10 = tf.keras.datasets.cifar10
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xTrain = xTrain / 255
xTest = xTest / 255

# Setting up the sequential model
model = tf.keras.models.Sequential()

# Adding the convolutional layers to the sequential model declared above
add_convolutional_to(model)

# Adding the fully connected layers as well as the final layer
add_connected_layer_to(model)

# Setting up the model's evaluation characteristics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Running the model and saving the results in a variable so they can be plotted later
record = model.fit(xTrain, yTrain, epochs=15,
                   validation_data=(xTest, yTest), validation_freq=1)

# Plotting the model training data
plot_model_data(record, 'loss')
plot_model_data(record, 'val_loss')

# Running the trained model on test data and saving results to a variable so the results can be plotted later
predictions = test_model(model)
usePredictions = [np.argmax(k) for k in predictions]

# Creating a confusion matrix and then plotting it in a visual format with Seaborn
cm = tf.math.confusion_matrix(labels=yTest, predictions=usePredictions)
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
