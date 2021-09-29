# import matplotlib.pyplot for graphing
import matplotlib.pyplot as plt

# import tensorflow for creating the neural network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# use keras to import the CIFAR-10 dataset
from keras.datasets import cifar10

# number of Epochs when training the neural network. (can be personally modified before running)
NUM_EPOCHS = 150

# from the CIFAR-10 dataset, load in the training images and testing images
(trainImages, trainLabels), (testImages, testLabels) = cifar10.load_data()

# divide the images by 255 to get decimals between 0 and 1
trainImages = trainImages/255
testImages = testImages/255

# creates the neural network model using Keras
model = keras.Sequential([
    Conv2D(16, 3, padding = 'same', activation = 'relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding = 'same', activation = 'relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding = 'same', activation = 'relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(units = 128, activation = 'relu'),
    Dense(units = 10, activation = "softmax")
])

# prepares the model for training using the adam optimizer, the type of loss being used, lastly show how we meant to evaluate the performance
model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])

# run our training using the train images and labels. Run 10 epochs (how many times the model will train for), shuffle the data after each time,
# display animated progress
history = model.fit(x = trainImages, y = trainLabels, epochs = NUM_EPOCHS, shuffle = True, verbose = 1)

# evaluate the trained model on the testing images and labels. record the number of losses and accuracy
finalLoss, finalAcc = model.evaluate(testImages, testLabels)

# print out the final accuracy
print("Final Accuracy: ", finalAcc)

# creating a list of all epochs (makes of 1 to the number of epochs with an increment of 1)
epochCount = range(1, NUM_EPOCHS + 1, 1)

# plot the number of losses for each epoch
plt.plot(epochCount, history.history['loss'], 'm-')

# plot the accuracy at each epoch
plt.plot(epochCount, history.history['accuracy'], 'g-')

# define the legend and labels of the plot
plt.legend(['Number of Losses', 'Accuracy of Epoch'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss and Accuracy')

# display the final plot
plt.show()
