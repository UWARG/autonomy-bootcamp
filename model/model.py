import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Convolution2D, Activation, Dropout, Dense
import numpy as np
import matplotlib.pyplot as plt
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# Load data
# Code for unpacking data, provided by the CIFAR-10 homepage
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batchNames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

# Training set
xTrainRaw = np.empty((0,3072),int)
yTrainRaw = np.empty((0))

# Loop through the data batches and create one large training set
for name in batchNames:
    
    trainBatch = unpickle("../data/" + name)
    xTrainRaw = np.append(xTrainRaw, trainBatch[b'data'], axis=0)
    yTrainRaw = np.append(yTrainRaw, trainBatch[b'labels'], axis=0)

test = unpickle('../data/test_batch')
print(xTrainRaw.shape)

# Test set
xTestRaw = test[b'data']
yTestRaw = np.asarray(test[b'labels'])

# Perform data transformation
# Convert from 
xTrain = xTrainRaw.reshape(50000, 32,32,3)
xTest = xTestRaw.reshape(10000,32,32,3)

yTrain = yTrainRaw
yTest = yTestRaw

# Normalize data between 0 and 1
# Max-min normalization
xTrain= xTrain / 255.0
xTest = xTest / 255.0


# Build the model
# I used the same architecture from https://www.tensorflow.org/tutorials/images/cnn
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10))
model.summary()

# Train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(xTrain, yTrain, epochs=10, 
                    validation_data=(xTest, yTest))


# Plot model accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.ylim([0.5, 1])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("../plots/model-performance.png")
testLoss, testAcc = model.evaluate(xTest,  yTest, verbose=2)
