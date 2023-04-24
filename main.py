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
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

def load_data():
    (training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
    training_images, testing_images = training_images/255, testing_images/255
    return (training_images, training_labels), (testing_images, testing_labels)

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(32,32,3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3,3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, training_data, validation_data, batch_size, epochs):
    history = model.fit(training_data[0], training_data[1], batch_size=batch_size, epochs=epochs, validation_data=validation_data)
    return history

def plot_loss(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(train_loss))
    plt.plot(epochs, train_loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_accuracy(history):
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(train_acc))
    plt.plot(epochs, train_acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Load data
training_data, testing_data = load_data()

# Create model
model = create_model()

# Train model
history = train_model(model, training_data, testing_data, batch_size=128, epochs=10)

# Evaluate model
loss, accuracy = model.evaluate(testing_data[0], testing_data[1])
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Plot results
plot_loss(history)
plot_accuracy(history)

model.save('image_classifier.model')

