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
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPool2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np
# Your working code here

# Load in dataset
def data_prep():

    (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()
    # Divide all rgb values by 255 to normalize data
    xTrain = xTrain/255.0
    xTest = xTest/255.0
    # Flatten extra dimension for y data since they are loaded as a (samples, 1) shaped tensor
    print("y_train.shape", yTrain.shape)
    yTrain = yTrain.flatten()
    yTest = yTest.flatten()

    return xTrain, yTrain, xTest, yTest

# Model input layer
def create_model(shape):

    i = Input(shape = shape)

    # First convolution layer
    x = Conv2D(32, (3,3), padding='same', activation='relu')(i)
    #  Batch normalization recenters and rescales data to help train faster
    x = BatchNormalization()(x)
    x = Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    # Max pool takes max value of each 2v2 square of each feature map to shink features which
    # This helps the model from overfitting by making the model more translationally invariant
    x = MaxPooling2D((2,2))(x)

    # Second convolution layer
    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    # Third convolution layer
    x = Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    # Dropout layer drops 30% of weights randomly to prevent overfitting
    x = Dropout(0.3)(x)

    # Global max pool takes max of feature maps for classficiation
    x = GlobalMaxPool2D()(x)

    #Dense layer with softmax activation for classification
    x = Dense(10, activation='softmax')(x)

    model = tf.keras.models.Model(i, x)
    
    return model

def train_model(model):

    # Compile model
    model.compile(optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])

    # Stops training if it has no improved validation accuracy for 10 epochs
    # Restores weights from epoch with greatest validation accuracy
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10, restore_best_weights=True
    )

    # Data generator to randomly shift images to prevent overfitting and teach the model to generalize mroe
    batchSize = 32
    dataGenerator = ImageDataGenerator(width_shift_range= 0.1, rotation_range= 20, height_shift_range= 0.1, horizontal_flip=True)
    trainGenerator = dataGenerator.flow(xTrain, yTrain, batchSize)
    stepsPerEpoch = xTrain.shape[0]//batchSize

    # Train model
    r = model.fit(trainGenerator, validation_data=(xTest, yTest), steps_per_epoch=stepsPerEpoch, epochs= 150, callbacks = [callback])

    return r

def plot_training_graphs(r):
    # Plot validation loss
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    # Plot validation accuracy
    plt.plot(r.history['accuracy'], label='acc')
    plt.plot(r.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()


if __name__ == "__main__":
        
    xTrain, yTrain, xTest, yTest = data_prep()
    model = create_model(xTrain[0].shape)
    r = train_model(model)
    plot_training_graphs(r)