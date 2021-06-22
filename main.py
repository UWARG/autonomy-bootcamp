import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(trainImages, trainLabels), (testImages, testLabels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
trainImages, testImages = trainImages/255.0, testImages/255.0

# We create a convolutional base using a stack of Conv2D and MaxPooling2D layers.
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten the 3D tensor to 1D
model.add(layers.Flatten())

# Add dense layers on top
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

EPOCHS = 10
# Train the model
history = model.fit(trainImages, trainLabels, epochs=EPOCHS,
                     validation_data=(testImages, testLabels))

# Make a plot of loss and validation loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.5, 1.5])
plt.legend(loc='lower right')

# Save file to plot.png
plt.savefig("plot.png")
