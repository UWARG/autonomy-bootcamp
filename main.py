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
        3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), activation='relu'))
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
        if entered < 0:
            break
        print(CLASS_NAMES[np.argmax(predictions[entered])])
        plt.imshow(xTest[entered])
        plt.show(block=False)
        plt.pause(2)
        plt.close('all')
    return predictions


cifar10 = tf.keras.datasets.cifar10
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xTrain = tf.keras.utils.normalize(xTrain, axis=1)
xTest = tf.keras.utils.normalize(xTest, axis=1)
model = tf.keras.models.Sequential()
add_convolutional_to(model)
add_connected_layer_to(model)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
record = model.fit(xTrain, yTrain, epochs=5,
                   validation_data=(xTest, yTest), validation_freq=1)
"""
plot_model_data(record, 'loss')
plot_model_data(record, 'val_loss')
"""
predictions = test_model(model)
usePredictions = [np.argmax(k) for k in predictions]
cm = tf.math.confusion_matrix(labels=yTest, predictions=usePredictions)
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
