# Imports
import itertools
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import activations

#load dataset
cifar = tf.keras.datasets.cifar10

#Constants
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
NUM_CLASSES = 10
EPOCHS = 25

def load_dataset():
    '''
    Loading the CIFAR-10 dataset and splitting into training, validation and test.
    '''
    # Load cifar10 dataset from Keras
    (xTrain, yTrain), (xTest, yTest) = cifar.load_data()

    # convert to binary class matrix
    # an alternative to this is keeping them as is and using sparse categorical-cross entropy loss
    yTrain = to_categorical(yTrain)
    yTest = to_categorical(yTest)

    # normalize images by scaling values to be between [0,1]
    xTrain = xTrain / 255.0
    xTest = xTest / 255.0

    # set validation size
    VALID_SIZE = 8000

    # Could have done K Cross fold Validation, but instead split with respect to a fixed size (based on percentage of total data)
    xValid, yValid = xTrain[:VALID_SIZE], yTrain[:VALID_SIZE]
    xTrain, yTrain = xTrain[VALID_SIZE:], yTrain[VALID_SIZE:]

    return (xTrain, yTrain), (xValid, yValid), (xTest, yTest)

# Visualize first 9 images
"""for i in range(9):
  plt.subplot(330+1+i)
  plt.imshow(xTrain[i])

plt.show()"""

def net():
  """
  Create Model Arcitecture. 4 Conv layers, 2 Dense layers and no max pooling
  For higher accuracy can transfer learn (eg. ResNet50) however, this is a simple CNN structure.
  The architecture is based on this paper: https://arxiv.org/pdf/1412.6806.pdf
  Max-pooling between layers are replaced by a convolutional layers with increased stride
  """

  model = tf.keras.models.Sequential()

  # Relu activation to avoid vanishing gradient problem
  model.add(tf.keras.layers.Conv2D(input_shape=xTrain[0,:,:,:].shape, activation= 'relu', filters=96, kernel_size=(3,3)))
  model.add(tf.keras.layers.Conv2D(filters=96, activation= 'relu', kernel_size=(3,3), strides=2))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.Conv2D(filters=192, activation= 'relu',kernel_size=(3,3)))
  model.add(tf.keras.layers.Conv2D(filters=192, activation= 'relu',kernel_size=(3,3), strides=2))
  model.add(tf.keras.layers.Dropout(0.5))

  # Flatten before passing into dense layers
  model.add(tf.keras.layers.Flatten())

  # Batch normalization to standardize the inputs
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Dense(256, activation= 'relu',))

  #Softmax activation on last layer to create probability distribution vector for each class
  model.add(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"))

  return model


def plot_results(history):
    '''
    Creates a plot of the training / validation loss and accuracy
    '''
    # create two subplots (one for loss and one for accuracy)
    fig, (loss, acc) = plt.subplots(2)
    fig.suptitle('Loss and Accuracy Plots')

    # plotting the training & validation loss per epoch
    loss.plot(history.history['val_loss'], color='red', label='Validation loss')
    loss.plot(history.history['loss'], color='green', label='Training loss')

    #set labels and legend
    loss.set_xlabel('Epoch')
    loss.set_ylabel('Loss')
    loss.legend(loc='upper right')

    # plot training & validation accuracy per epoch
    acc.plot(history.history['accuracy'], color='blue', label='training accuracy')
    acc.plot(history.history['val_accuracy'], color='red', label='validation accuracy')

    #set labels and legend
    acc.set_xlabel('Epoch')
    acc.set_ylabel('Accuracy')
    acc.legend(loc='lower right')

    plt.show()

def main():
    # prepare and load cifar10 data
    (xTrain, yTrain), (xValid, yValid), (xTest, yTest) = load_dataset()

    model = net()

    # Prepare model for training (categorial_crossentropy since data is represented as a binary class matrix)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # set batch size to be used in model training
    BATCH_SIZE=128

    # train model
    history = model.fit(xTrain, yTrain, batch_size=BATCH_SIZE,
                      epochs=EPOCHS, validation_data=(xValid, yValid))

    # plot the results
    plot_results(history)
    
    # evaluate model on test data which the model has never seen and print the results
    testLoss, testAcc = model.evaluate(xTest, yTest)
    print('\nTEST LOSS RESULT: ' + str(testLoss))
    print('TEST ACC RESULT: ' + str(testAcc)) 


main()











