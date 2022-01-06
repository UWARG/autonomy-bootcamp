"""
This is a starter file to get you going. You may also include other files if you feel it's necessary.

Make sure to follow the code convention described here:
https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions

Hints:
* The internet is your friend! Don't be afraid to search for tutorials/intros/etc.
* We suggest using a convolutional neural network.
* TensorFlow Keras has the CIFAR-10 dataset as a module, so you don't need to manually download and unpack it.
"""

# Import whatever libraries/modules you need

import sys
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from keras.layers import BatchNormalization
import matplotlib.pyplot as pyplot

# Your working code here

# loading the train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    # returning the train and test data
    return trainX, trainY, testX, testY


# scaling the pixels
def prep_pixels(train, test):
    # converting from integers to floats
    trainNorm = train.astype('float32')
    testNorm = test.astype('float32')
    # normalizing pixel colours to be between 0 and 1
    trainNorm = trainNorm/255.0
    testNorm = testNorm/255.0
    # return normalized images
    return trainNorm, testNorm


# define cnn model
def define_model():
    # creating a sequential model
    model = Sequential()

    # 1st conv block
    # 2-dimensional convolution layer to capture low level features
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    # to speed up and stablize the training process
    model.add(BatchNormalization())
    # max pooling layer to progressively reduce spacial size of the image as well as to act as a noise suppressant
    model.add(MaxPooling2D((2, 2)))
    # dropout layer to minimize over fitting
    model.add(Dropout(0.2))

    # 2nd conv block
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    # 3rd conv block
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    # flatten to convert data into a 1D array
    model.add(Flatten())
    # dense layers used to classify image based on output from convolutional layers.
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))

    # compile model
    opt = SGD(lr=0.005, decay=1e-6, momentum=0.9)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


# plot diagnostic learning curves
def summarize_diagnostics(trainedModel):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(trainedModel.history['loss'], color='blue', label='training loss')
    pyplot.plot(trainedModel.history['val_loss'], color='orange', label='validation loss')
    pyplot.legend(loc="upper right")

    #saving the plot
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


# run the test harness for evaluating a model
def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = define_model()
    # fit model
    trainedModel = model.fit(trainX, trainY, epochs=40, validation_data=(testX, testY), validation_split=0.2, verbose=1)
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=1)
    print('final accuracy: %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(trainedModel)


# entry point, run the test harness
run_test_harness()