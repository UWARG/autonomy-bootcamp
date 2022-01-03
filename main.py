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

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# Your working code here

# -- training and testing model --
# load data 
(xTrain, yTrain), (xTest,yTest) = datasets.cifar10.load_data()

# reshape to 1D
yTrain = yTrain.reshape(-1,) 

# normalize data to be between 0 and 1
xTrain = xTrain / 255.0
xTest = xTest / 255.0

# make convolutional network
cnn = models.Sequential([
    # convolution + pooling layer 1
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # convolution + pooling layer 2
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # convolution + pooling layer 3
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # flatten
    layers.Flatten(),

    # fully connected layers
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# compile model
cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train cnn with training data
history = cnn.fit(xTrain, yTrain, epochs=10, validation_data=(xTest, yTest))

# evaluate model with testing data
cnn.evaluate(xTest, yTest)


# -- plotting data -- 

# divide plot into two subplots
fig, (ax1, ax2) = plt.subplots(1, 2)
# title
fig.suptitle('Model Performance')

# plot training accuracy
ax1.plot(history.history["accuracy"])
# plot validation accuracy
ax1.plot(history.history["val_accuracy"])
ax1.set_title("Accuracy")
ax1.set(ylabel='Accuracy', xlabel='Epochs')
ax1.legend(["Training", "Validation"])

# plot training loss
ax2.plot(history.history["loss"])
# plot validation loss
ax2.plot(history.history["val_loss"])
ax2.set_title("Loss")
ax2.set(ylabel='Loss', xlabel='Epochs')
ax2.legend(["Training", "Validation"])

plt.show()

