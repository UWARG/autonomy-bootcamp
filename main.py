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
import numpy as np
import matplotlib.pyplot as plt

# Your working code here
# Author - Tanishq Koul

def get_data():
    """
    Loads and normalizes CIFAR-10 Data
    Returns
    -------
    tuple<tuple<np.array, np.array>, tuple<np.array, np.array>>
        Returns CIFAR-10 training and validation normalized from 0-1
    """

    # load CIFAR-10 dataset
    (fitData, fitLabels), (valData, valLabels) = tf.keras.datasets.cifar10.load_data()

    # normalize data
    fitData = fitData / 255.0
    valData = valData / 255.0

    return (fitData, fitLabels), (valData, valLabels)

def get_model():
    """
    Creates CNN using Sequential Model
    
    Returns
    ------
    tf.keras.Model
        Returns a keras CNN model
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    return model

def plot_loss_accuracy(model_history):
    """
    Plots accuracy & loss of a model with matplotlib
    
    Params
    ------
    tf.keras.History
        The history object returned by training a model
    """

    plt.figure()

    #plots loss over epochs
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])

    plt.title('loss over epochs')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(['Training', 'Validation'])


    #plots accuracy over epochs
    plt.figure()

    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])

    plt.title('accuracy over epochs')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.legend(['Training', 'Validation'])

    plt.show()
 

def main():
    """
    main function for training model and call plot function
    """
    # get data
    (fitData, fitLabels), (valData, valLabels) = get_data()
    model = get_model()

    # fit data
    model.fit(fitData, fitLabels, epochs=10, validation_data=(valData, valLabels))

    #plot results
    plot_loss_accuracy(model.history)

main()