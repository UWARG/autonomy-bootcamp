#import required libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint

from sklearn.model_selection import train_test_split
import tensorflow as tf


# Functions

#function to unpickle CIFAR10 dataset
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#function to plot train and test accuracy for models based on history object
def plot_acc(history):

    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    #plt.rc('font', serif='Times')
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    accuracy = history.history['accuracy']
    valAccuracy = history.history['val_accuracy']
    loss = history.history['loss']
    valLoss = history.history['val_loss']
    epochs = range(len(accuracy)+1)
    epochs = epochs[1:]

    plt.figure(figsize=(10,6))
    plt.plot(epochs, accuracy, 'r', label='Training accuracy')
    plt.plot(epochs, valAccuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()
    plt.savefig('./accuracy_plot.png')

    plt.figure(figsize=(10,6))
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, valLoss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig('./loss_plot.png')
    plt.show()

if __name__ == "__main__":
    ## unpickle meta data to find the classes of CIFAR10 dataset
    batchesMeta = unpickle('./cifar-10-batches-py/batches.meta')
    classes = batchesMeta[b'label_names']

    ## unpickle and prepare training data
    ## read and merge all train data into single train_data
    trainData = np.zeros([50000, 32, 32, 3], dtype=np.uint8)
    trainLabels = np.zeros([50000,], dtype=np.uint8)
    nofBatch = 5
    counter = 1
    while (counter<=nofBatch):
        filename = './cifar-10-batches-py/data_batch_' + str(counter)
        #read dictionary from file
        dataDict = unpickle(filename)    
        #extract data and their labels from dictionary
        data = dataDict[b'data']   
        labels = np.array(dataDict[b'labels'])    
        # append data into single array
        trainData[(counter-1)*10000:counter*10000] = (data.reshape(10000, 3,32,32)).transpose(0, 2, 3, 1)
        trainLabels[(counter-1)*10000:counter*10000] = labels
        counter += 1


    ## unpickle and prepare test data
    ## fetch test data from test_batch
    testFilename = './cifar-10-batches-py/test_batch'
    #read test dictionary from file
    testDict = unpickle(testFilename)

    #extract test data and the labels
    testData = (testDict[b'data'].reshape(10000, 3,32,32)).transpose(0, 2, 3, 1)
    testLabels = np.array(testDict[b'labels'])


    #prepare data for training CNN based model
    #data normalization
    trainDataNorm = trainData / 255.
    testDataNorm = testData / 255.

    #one hot encoding of labels
    trainLabelsEnc = to_categorical(trainLabels)
    testLabelsEnc = to_categorical(testLabels)

    #split training data into train and validation data
    xTrain, xValid, yTrain, yValid  = train_test_split(trainDataNorm, trainLabelsEnc, train_size=0.8, random_state=42)
    xTest, yTest = testDataNorm, testLabelsEnc

    #CNN Model for image classifier
    inputShape = xTrain.shape[1:]
    nofClasses = yTrain.shape[1]
    nofEpochs = 10
    batchSize = 64
    lossFunction = 'categorical_crossentropy'
    learningRate = 1e-3

    #create CNN based model for image classifier
    imageClassifier = Sequential([
                    Conv2D(filters=64, kernel_size=3, input_shape = inputShape, activation='relu'),
                    MaxPooling2D(2,2),
                    Dropout(0.1),
                    Conv2D(filters=64, kernel_size=3, activation='relu'),  
                    MaxPooling2D(2,2),
                    Dropout(0.1),
                    Conv2D(filters=32, kernel_size=3, activation='relu'),  
                    MaxPooling2D(2,2),
                    Dropout(0.2),
                    Flatten(),
                    Dense(64, activation='relu'),  
                    Dropout(0.2),    
                    Dense(32, activation='relu'),  
                    Dropout(0.2),
                    Dense(nofClasses, activation='softmax') 
    ])

    #compile the CNN model and print summary of model
    imageClassifier.compile(loss=lossFunction, optimizer=Adam(lr=learningRate), metrics=['accuracy'])
    imageClassifier.summary()

    #train the model using training data and validation data for validation purpose
    history = imageClassifier.fit(xTrain, yTrain, batch_size=batchSize, 
                    epochs=nofEpochs, validation_data=(xValid, yValid),
                    verbose=2, shuffle=True)


    #Use the trained model to predict unseen test data and print accuracy result
    score = imageClassifier.evaluate(xTest, yTest, verbose=1)
    print("Accuracy: ", round(score[1],4))

    #draw and save train and test accuracy for model
    plot_acc(history)

