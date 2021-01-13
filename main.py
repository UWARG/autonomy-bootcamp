import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Get data
(trainData, trainLabels), (testData, testLabels) = cifar10.load_data()

# Normalize image data
trainData = trainData / 255
testData = testData / 255

# Build network
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape = [32, 32, 3]))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2, padding='valid'))
# Drop outer edges with probability
model.add(tf.keras.layers.Dropout(0.5))
# Serialize 
model.add(tf.keras.layers.Flatten())
# Fully-connected layer
model.add(tf.keras.layers.Dense(units = 128, activation='relu'))
# Last layer for classification
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

#model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # Dense (fully-connected) layer with 128 nodes
#model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # Last layer for classification

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track

epochs = 10

# Train network
performanceHistory = model.fit(trainData, trainLabels, epochs=epochs)

# Display graph of accuracy and loss vs epoch
epochRange = range(1, epochs + 1)
plt.plot(epochRange, performanceHistory.history['accuracy'])
plt.plot(epochRange, performanceHistory.history['loss'])
plt.title('Model')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Loss'], loc='upper left')
plt.show()

# Performance on test data
val_loss, val_acc = model.evaluate(testData, testLabels)
print(val_loss)  # model's loss (error): ~0.91
print(val_acc)  # model's accuracy: ~0.70

# Sources
# https://kgptalkie.com/2d-cnn-in-tensorflow-2-0-on-cifar-10-object-recognition-in-images/
# https://www.cs.toronto.edu/~kriz/cifar.html

# TODO: Unrelated read
# https://www.st.com/resource/en/reference_manual/dm00091010-stm32f030x4x6x8xc-and-stm32f070x6xb-advanced-armbased-32bit-mcus-stmicroelectronics.pdf
# http://www.micromouseonline.com/2016/02/06/pwm-basics-on-the-stm32-general-purpose-timers/
