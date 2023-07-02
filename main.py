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

# Making the necessary imports here
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau

"""
  Load the dataset from tensorflow datasets and split into images and labels
"""
BATCH_SIZE = 32

(train_imgs, train_labels), (test_imgs, test_labels) = datasets.cifar10.load_data()

# Define a CNN in PyTorch
def get_model():
  """
    Returns a simple CNN model for classifying CIFAR-10
    Images

    Uses 6 Convolutions of 2x2 kernels
    Activation = Relu

    The classifier head is a 3 layer MLP
    Of output dimensions 512, 256 and 10

    This function simply returns this tensorflow model
  """
  model = Sequential([
      # Make a convolutional layer of kernel size 2x2 with relu activation and 64 output filters
      Conv2D(32, (2, 2), activation = 'relu', kernel_initializer = 'he_uniform', input_shape = (32, 32, 3)),
      # Apply batch normalization for fast convergence
      BatchNormalization(),
      Conv2D(64, (2, 2), activation = 'relu'),
      # Add a Max Pooling layer that takes the maximum of a window of 2x2 kernels - halves the dimension as well
      MaxPool2D(),

      Conv2D(64, (2, 2), kernel_initializer = 'he_uniform',activation = 'relu'),
      BatchNormalization(),
      Conv2D(64, (2, 2), kernel_initializer = 'he_uniform',activation = 'relu'),
      Dropout(0.2),
      MaxPool2D(),

      Conv2D(128, (2, 2), kernel_initializer = 'he_uniform', activation = 'relu'),
      BatchNormalization(),
      Conv2D(128, (2, 2), kernel_initializer = 'he_uniform', activation = 'relu'),
      # Use dropout for preventing overfitting
      Dropout(0.2),

      # Convert the image features into a single vector for passing through an MLP
      Flatten(),

      # MLP layers
      Dense(512, 'relu'),
      Dropout(0.2),
      Dense(256, 'relu'),
      Dense(10, 'softmax')
  ])
  return model

model = get_model()

# Just checking the model parameters and structure
model.summary()

# Compile the model with loss function, and metrics that we shall track during training
model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False), # Softmax is already applied so logits = False
    metrics = ['accuracy']
)

def train_model(model):
  """
    Train the model with Adam optimizer, and ReduceLROnPlateau lr callback
    The model is trained for 25 epochs

    Returns the history which contains the training loss, accuracy and
    validation loss and accuracy
  """
  EPOCHS = 25
  # Reduce LR for increasinging val accuracy
  reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 0)

  # Fit the model on training data, and also use the reduce_lr callback to optimize the val_accuracy
  # We train for 25 epochs
  history = model.fit(train_imgs, train_labels, epochs = EPOCHS,
            validation_data = (test_imgs, test_labels), callbacks = [reduce_lr])
  return history

history = train_model(model)

def plot_model_training(history):
  """
    Plots out the Training and Validation Losses of the model
    Also plots the training and validation accuracy of the model

    Does not return anything.
  """
  plt.plot(history.history['loss'], label = 'Model Training Loss')
  plt.plot(history.history['val_loss'], label = 'Model Validation Loss')
  plt.legend(["Training Loss", "Val Loss"])

  plt.plot(history.history['accuracy'], label = 'Model Training Accuracy')
  plt.plot(history.history['val_accuracy'], label = 'Model Validation Accuracy')
  plt.legend(["Training Accuracy", "Val Accuracy"])
