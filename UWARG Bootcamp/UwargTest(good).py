#Made by Soumav Maiti
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
#gets data
data = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = data.load_data()

imagetype_names= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 
            'frog', 'horse', 'ship', 'truck']

# Normalize data between 0-1
train_images = train_images / 255
test_images = test_images / 255

#model builder
model = keras.Sequential([
  keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(10, activation="softmax")
])

#model compiler, stores history of model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10)

#sets the accuracy and loss of model based on passed tests
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Tested accuracy: ", test_acc)

# Get training and test loss histories
training_loss = history.history['loss']
test_acc = history.history['accuracy']

#Number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r-')
plt.plot(epoch_count, test_acc, 'b-')
plt.legend(['Training Loss', 'Test Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.show()
