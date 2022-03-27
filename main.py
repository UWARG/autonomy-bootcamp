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
import numpy as np
import matplotlib.pyplot as plt

# Your working code here

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Pre-process training and testing set
# Scale image pixel values from 0-255 to 0-1
train_images = train_images/255.0
test_images = test_images/255.0

# To build neural network we need to configure the layers of the model then compile the model

# Setting up layers of the model
# Sequential model means single input and output so in this case the input is an image and the output is a label
model = keras.Sequential([

    # Idea behind a convolutional neural network is that we filter the images before training the deep neural network
    # This makes it so the features of the image are enhanced by the filters and can be used to identify corresponding outputs
    # After the image is filtered it is pooled which groups up pixels in the image then filters them down to a subset 
    # This lowers the resolution of the image but maintains the important features from the filtering 

    # The idea is that random filters pass over the image to enhance different features then these features are matched with the image output
    # Then over time the filters that give us the features that most match the corresponding output is learned
    # This is called feature extraction

    # We can stack convolutional layers to break down the image even further and learn from more abstract features
    # For the first one, it will generate 64 filters and multiply all of them across the image so then each epoch it will figure out which filters gave the best features to best match the image with its corresponding output
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape = (32,32,3)),
	keras.layers.MaxPooling2D((2, 2)),

	keras.layers.Conv2D(64, (3, 3), activation='relu'),
	keras.layers.MaxPooling2D((2, 2)),

	keras.layers.Conv2D(128, (3, 3), activation='relu'),
	keras.layers.MaxPooling2D((2, 2)),

    # Flattening turns our array into a 1D array to feed into the dense layers
    # It is only meant to reformat the data
	keras.layers.Flatten(),

    # These last layers are what does the actual classification
	keras.layers.Dense(128, activation='relu'),
	keras.layers.Dense(10, activation='softmax')
])

# Compiling the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Start the training of the model and store its history in model_history
model_history = model.fit(train_images, train_labels, epochs = 10, validation_data = (test_images, test_labels))

# Plot the training and validation loss over epochs
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Training and Validation Loss over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc = 'upper left')
plt.show()
