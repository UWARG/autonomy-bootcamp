# Import necessary libraries
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models

# Loading data
(trainImage, trainLabel), (testImage, testLabel) = datasets.cifar10.load_data()

# Normalize pixel values
trainImage, testImage = trainImage/255.0, testImage/255.0

# Available Class Names
className = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

# Create and Train the model
# Define Sequential model 
model = models.Sequential()
# Create a 2D Convolutional Layer with 32 nodes, with a kernel size of (3,3), 
# activation function relu and input shape of (32, 32, 3)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# Create a MaxPooling 2D Layer of pool size (2,2)
model.add(layers.MaxPooling2D((2, 2)))
# Create a 2D Convolutional Layer with 64 nodes, with a kernel size of (3,3), 
# activation function relu
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Create a MaxPooling 2D Layer of pool size (2,2)
model.add(layers.MaxPooling2D((2, 2)))
# Create a 2D Convolutional Layer with 64 nodes, with a kernel size of (3,3), 
# activation function relu
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Flattens the output into 1 dimension
model.add(layers.Flatten())
# Add dense layers
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# Fit the model to our training data, and validate it with our test data. 
history = model.fit(trainImage, trainLabel, epochs=10, 
                    validation_data=(testImage, testLabel))

# Plot the accuracy
plt.plot(history.history['accuracy'], label ='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
# Evaluate the accuracy of the model
test_loss, test_acc = model.evaluate(testImage,  testLabel, verbose=2)
