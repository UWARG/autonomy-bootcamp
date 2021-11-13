from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt

# load data - 'test' items will be used for validation data
(trainImages, trainLabels), (testImages, testLabels) = cifar10.load_data()

# reshape tensors, reduce magnitude of elements in tensors
trainImages = trainImages.reshape((50000, 32, 32, 3)) 
trainImages = trainImages.astype('float32')/255 
testImages = testImages.reshape((10000, 32, 32, 3)) 
testImages = testImages.astype('float32')/255

# convert tensors to categorical data (from values of 1 to 10 to vectors with binary values indicating category)
trainLabels = to_categorical(trainLabels) 
testLabels = to_categorical(testLabels)

network = models.Sequential() # construction of neural network
network.add(layers.Conv2D(128, (3,3), activation = 'relu', input_shape = (32, 32, 3))) # input layer, convolution for feature extraction
network.add(layers.MaxPool2D((2,2))) # max pool to filter/enhance feature extractions
network.add(layers.Conv2D(128, (3,3), activation = 'relu'))
network.add(layers.MaxPool2D((2,2)))
network.add(layers.Conv2D(128, (3,3), activation = 'relu'))
network.add(layers.MaxPool2D((2,2)))
network.add(layers.Flatten()) # flatten for tensors to be moved into dense layers
network.add(layers.Dense(64, activation = 'relu'))
network.add(layers.Dense(64, activation = 'relu'))
network.add(layers.Dense(10, activation = 'softmax')) # softmax activation for multiclass single-label classification

# compile the network 
network.compile(optimizer = 'rmsprop', # a standard optimizor
                loss = 'categorical_crossentropy',  # loss function for multiclass single-label classification
                metrics = ['accuracy'])

# Fit the data through the network, 
data = network.fit(trainImages, trainLabels, epochs = 8, batch_size = 75, validation_data= (testImages, testLabels)) # use 'testImages' as validation data

# construct pyplot to compare training accuracy and validation accuracy
plt.plot(range(1, len(data.history['accuracy']) + 1), data.history['accuracy'], label = 'Training Accuracy')
plt.plot(range(1, len(data.history['val_accuracy']) + 1), data.history['val_accuracy'], label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

