# The CIFAR-10 Dataset has 60,000 images and 10 classes
# There are 50,000 training images and 10,000 test images

# Libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load the CIFAR-10 dataset, returns 4 numpy arrays
(images_train, labels_train), (images_test, labels_test) = datasets.cifar10.load_data()

# For example, images_train is a numpy array containing the training images from the dataset.
# It has a shape of (50000, 32, 32, 3)
# 50,000 is the number of training samples
# each sample is a 32 x 32 image
# and there are 3 color channels (red, green, and blue)
print("Training Image Details:", images_train.shape)
# labels_train is a numpy array containing the corresponding labels for the training images. 
# It has a shape of (50000)
# each value is an integer representing the class label for the corresponding image in images_train
print("Training Label Details:", labels_train.shape)
# The same applies for the test images and labels
print("Test Image Details:", images_test.shape) 
print("Test Label Details:", labels_test.shape) 

print("Numpy array representation of the first training image:")
print(images_train[0])
print()

print("Visual representation of the first training image:")
plt.imshow(images_train[0])

# Quick code snippet for data exploration

# Turn numpy array of labels into a one dimensional array.
labels_train = labels_train.reshape(-1,)
labels_test = labels_test.reshape(-1,)

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# x is a numpy array of training images
# y is the corresponding numpy array of training labels
# index specifies which training image to plot
def plot_sample(x, y, index):
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])

# Now we can easily plot some images
plot_sample(images_train, labels_train, 175)
plot_sample(images_test, labels_test, 12)
plot_sample(images_test, labels_test, 28)

# Normalize the data by dividing each pixel value in the images by 255
# obtaining a number ranging from 0 to 1

# Example:
print("Normalized numpy array of the first image in training set")
print(images_train[0] / 255)

images_train = images_train / 255
images_test = images_test / 255

# Building a Convolutional Neural Network
cnn = models.Sequential([
    # CNN
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Dense Layer
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Train the model
history = cnn.fit(images_train, labels_train, validation_data=(images_test, labels_test), epochs=8)

# Graph for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy Plot')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Graph for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss Plot')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model's accuracy on test set
print("Accuracy:")
cnn.evaluate(images_test,labels_test)

# Predict labels of training images
probabilities = cnn.predict(images_test)

# The predict() function returns an array of probabilities 
# indicating the likelihood of each label for the test image at that index
print("Probability Array:")
print(probabilities)

# The argmax function returns the index of the largest number in a list
print("Arg Max Example:")
print(np.argmax([2, 7, 5]))

predictions = [np.argmax(p) for p in probabilities]

# Extra: A function that determines if a model correctly predicted the label
# for a specific image

def correct(index):
    if (predictions[index] == labels_test[index]):
        print("Correct Prediction: ", classes[predictions[index]])
    else:
        print("Incorrect Prediction: ", classes[predictions[index]])
        print("Actual: ", classes[labels_test[index]])

correct(12)
correct(0)
correct(150)
correct(28)
