# Importing assets
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import optimizers, losses
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, MaxPooling2D, Flatten, Conv2D

"""
Loading CIFAR-10 images via Keras datasets
In our case, 50k training pairs and 10k testing pairs
"""
(trainImgs, trainLabs), (testImgs, testLabs) = cifar10.load_data()

# To normalize image pixel values to a floating point value in range [0,1], we true divide by 255.0
trainImgs = trainImgs/255.0
testImgs = testImgs/255.0

# We define our model to be sequential - layer stack of which contains one I/O tensor per layer
model = Sequential()

"""
We define our CNN model - very common architecture
The first two additions to our model are that of 2D convolutional layers. The first layer accepts an input shape
definition - 32x32 RGB pictures. Rectified linear unit activation function is used - f(x) = {x if x > 0, 0 otherwise
"""
model.add(Conv2D(32, activation='relu', kernel_size=(3, 3), input_shape=(32, 32, 3)))
model.add(Conv2D(32, activation='relu', kernel_size=(3, 3)))

# Maxpooling used to down sample our data by "summarizing" a set group of values by its greatest value
model.add(MaxPooling2D((2, 2), padding='same'))

# Dropout layer added in attempt to prevent overfitting - randomly destroys some outgoing node-edge data
model.add(Dropout(0.2))

# Repeat
model.add(Conv2D(64, activation='relu', kernel_size=(3, 3)))
model.add(Conv2D(64, activation='relu', kernel_size=(3, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(128, activation='relu', kernel_size=(3, 3)))
model.add(Conv2D(128, activation='relu', kernel_size=(3, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.4))

# Converting matrix data to 1D array, then passing into dense layers
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10))

"""
We compile our model.
AdaM used considering we want to use stochastic gradient descent for obj. func optimization
"""
model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

history = model.fit(trainImgs, trainLabs, epochs=10,
                    validation_data=(testImgs, testLabs))

testLoss, testAcc = model.evaluate(testImgs, testLabs, verbose=1)


# PyPlot configuration
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss Comparison')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
