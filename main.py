import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def data_preprocess(pixelMax):
    """
    loads in CIFAR-10 data, normalizes the features, and one hot encodes the target variables

    Args:
        pixelMax: int
            indicates the maximum pixel value for normalization

    Returns:
        xTrain: numpy.ndarray
            train features
        yTrain: numpy.ndarray
            train targets
        xVal: numpy.ndarray
            validation features
        yVal: numpy.ndarray
            validation targets
    """
    (xTrain, yTrain), (xVal, yVal) = tf.keras.datasets.cifar10.load_data()

    # Normalize Data
    xTrain = xTrain / pixelMax
    xVal = xVal / pixelMax

    # One hot encode target variable
    yTrain = to_categorical(yTrain)
    yVal = to_categorical(yVal)

    return (xTrain, yTrain), (xVal, yVal)


def model_definition():
    """
    defines a CNN for image classfication on the CIFAR-10 Dataset

    Returns:
        tf.keras.Model
            defined CNN model

    model = Sequential(
        [
            Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 2), activation='relu'),
            MaxPooling2D(pool_size=(3, 3)),
            Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 2), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dense(units=10, activation='softmax')
        ]
    )
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10))
    return model


def model_training(model, xTrain, yTrain, xVal, yVal, epochs, loss, optimizer):
    """
    compiles model and then fits the training data on model. Returns training history.
    Uses (xTrain, yTrain) as training data. and (xVal, yVal) as a validation set.

    Args:
        model: tf.keras.Model
            model to train
        xTrain: numpy.ndarray
            train features
        yTrain: numpy.ndarray
            train labels
        xVal: numpy.ndarray
            validation features
        yVal: numpy.ndarray
            validation labels
        epochs: int
            number of training epochs
        loss: str
            name of loss function
        optimizer: str
            name of optimizer


    Returns:
        tf.keras.callbacks.History
            training history
    """
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    history = model.fit(
        x=xTrain,
        y=yTrain,
        validation_data=(xVal, yVal),
        epochs=epochs
    )
    return history


def plot_curves(history):
    """
    plots history of model training

    Args:
        history: tf.keras.callbacks.History
            model history that will be plotted

    Returns:
        None
    """
    plt.plot(history.history['loss'], label='Accuracy', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Accuracy', color='red')
    plt.legend()
    plt.title('Train and Validation Loss Curves')


if __name__ == '__main__':
    # CONFIGURATIONS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # CONSTANTS
    PIXEL_MAX = 255
    EPOCHS = 10
    LOSS = 'categorical_crossentropy'
    OPTIMIZER = 'adam'

    (xTrain, yTrain), (xVal, yVal) = data_preprocess(PIXEL_MAX)
    model = model_definition()
    history = model_training(model, xTrain, yTrain, xVal, yVal, EPOCHS, LOSS, OPTIMIZER)
    plot_curves(history)
