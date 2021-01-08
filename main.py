import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

files = []
data = []
imagesData = []
imagesLabels = []
trainData = np.empty((0, 32*32*3))
trainLabels = np.array([])
batches = 2 # Must be between 1 to 5 inclusive
# Note: Only batch 1 and 2 are saved on GitHub, download and unpack others if increasing this number

# Unpack image data
for i in range(0, batches):
    files.append("cifar-10-batches-py/data_batch_" + str(i + 1))
    data.append(unpickle(files[i]))
    imagesRawData = (data[i].get(b"data")) # 10000 rows, each row containing one image's data buffer
    # np.array required for keras model input
    imagesData = np.array(tf.keras.utils.normalize(imagesRawData, axis=1)) # Normalization between 0 to 1 inclusive
    trainData = np.concatenate((trainData, imagesData), axis=0)
    imagesLabels = np.array(data[i].get(b"labels")) # 10000 rows, each containing classification (0 to 9 inclusive)
    trainLabels = np.concatenate((trainLabels, imagesLabels))

# Image data to test on
testFile = "cifar-10-batches-py/test_batch"
testAllData = unpickle(testFile)
testRaw = testAllData.get(b"data")
testData = np.array(tf.keras.utils.normalize(testRaw, axis=1))
testLabels = np.array(testAllData.get(b"labels"))


# Debugging
#print(data[0].keys())

# Debugging
# Display first image of batch 1
# For making sure extraction was done correctly
# Note: This will display each of the colour channels in greyscale
#image1Serial = np.array(data[0].get(b"data")[0]) # Extract the raw image data from the dictionary
#image1RGB = image1Serial.reshape(32*3,32) # Reshape into 2D array
#print(image1RGB)
#plt.imshow(image1RGB,cmap="gray")
#plt.show()


# Build network
model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten()) # This flattens a 2D array, unneeded
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # Dense (fully-connected) layer with 128 nodes
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # Last layer for classification

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track

# Run network
model.fit(trainData, trainLabels, epochs=10) # Train it

val_loss, val_acc = model.evaluate(testData, testLabels) # See how well it performs on the test data
print(val_loss)  # model's loss (error): ~1.58
print(val_acc)  # model's accuracy: ~0.44


# https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/
# https://keras.io/api/datasets/mnist/#load_data-function
# https://www.cs.toronto.edu/~kriz/cifar.html

# TODO: Unrelated read
# https://www.st.com/resource/en/reference_manual/dm00091010-stm32f030x4x6x8xc-and-stm32f070x6xb-advanced-armbased-32bit-mcus-stmicroelectronics.pdf
# http://www.micromouseonline.com/2016/02/06/pwm-basics-on-the-stm32-general-purpose-timers/
