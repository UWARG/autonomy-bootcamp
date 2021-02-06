import pickle, os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10


HOME = "/home/aldec/data/03_uwarg"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
EPOCHS = 150
BATCH_SIZE = 100

"""
def loadFiles():
    def unpickle(file):
        print(file)
        with open(file, 'rb') as fo:
            rawDict = pickle.load(fo, encoding='bytes')
        # data is shaped (len, 32*32*3), needs to convert to (len, 32, 32, 3)
        batchData = rawDict[b"data"].reshape((len(rawDict[b"data"]), 3, 32, 32)).transpose(0, 2, 3, 1)
        batchLabels = rawDict[b"labels"]
        return batchData, batchLabels

    allData = []
    allLabels = []
    for batchNumber in range(1, 6):
        path = os.path.join(HOME, "data/data_batch_{}".format(batchNumber))
        data, labels = unpickle(path)
        # grouping all the data together
        allData.append(data)
        allLabels.append(labels)
    allData, allLabels = np.array(allData), np.array(allLabels)
    return allData.reshape([allData.shape[0]*allData.shape[1],32, 32, 3]), allLabels.reshape([allLabels.shape[0]*allLabels.shape[1]])
"""


def normalize(x):
    # so the image data is all 0-1, makes it easier on the model
    return (x-np.min(x))/(np.max(x)-np.min(x))


def oneHot(labelData, numClasses=len(CLASSES)):
    # prevent numerical associations between classes, ex. birds are like cats more than ship is like airplane
    blankArr = np.zeros((len(labelData), numClasses))
    for i in range(len(labelData)):
        blankArr[i][labelData[i]] = 1
    return blankArr


def model():
    # Functional API
    inputs = Input(shape=(32, 32, 3))
    # Conv2D is passing a kernel over the input and doing a multiplication, used to extract features
    conv1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    norm2 = layers.BatchNormalization()(conv1)
    drop1 = layers.Dropout(0.2)(conv1)
    # MaxPool2D is usually used to reduce the size of the image, kinda like a summary
    max_pool1 = layers.MaxPool2D((3, 3))(drop1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu')(max_pool1)
    norm2 = layers.BatchNormalization()(conv2)
    drop2 = layers.Dropout(0.2)(norm2)
    conv3 = layers.Conv2D(64, (3, 3), activation='relu')(drop2)
    flat = layers.Flatten()(conv3)
    # fully connected layers
    dense1 = layers.Dense(64, activation='relu')(flat)
    output = layers.Dense(10, activation='softmax')(dense1)

    cnn = Model(inputs=inputs, outputs=output)
    return cnn


def showLosses(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("/home/aldec/data/03_uwarg/FIGURE.png")
    plt.show()


if __name__ == "__main__":
    (xTrain, yTrain), (xVal, yVal) = cifar10.load_data()
    xTrain, yTrain = normalize(xTrain), oneHot(yTrain)
    # data augmentation, make model less vulnerable to new datasets
    augment = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=20, zoom_range=0.1, horizontal_flip=True)
    trainGenerator = augment.flow(xTrain, yTrain, batch_size=BATCH_SIZE)
    xVal, yVal = normalize(xVal), oneHot(yVal)
    cnn = model()
    print(cnn.summary())
    cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = cnn.fit(trainGenerator, validation_data=(xVal, yVal), epochs=EPOCHS)
    showLosses(hist)