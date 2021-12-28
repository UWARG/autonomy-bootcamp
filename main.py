import tensorflow
import matplotlib.pyplot as plt

from tensorflow.keras import datasets
from tensorflow.keras import layers
from tensorflow.keras import models

# Constants
IMAGE_SHAPE = (32, 32, 3)  # define image shape, will be used in CNN layers


def prepare_data():
    """
    Loads and prepares the CIFAR-10 dataset for the Convolutional Nerual Network (CNN) to use

    Parameters
    ----------
    None

    Returns
    -------
    list<np.ndarray>
        Returns a list containing x_train, x_test, y_train, y_test respectively
    """

    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    x_train = x_train / 255
    x_test = x_test / 255

    # Package into list and return
    dataset = [x_train, x_test, y_train, y_test]
    return dataset


def create_model():
    """
    Defines the CNN model layers and Hyperparameters

    Parameters
    ----------
    None

    Returns
    -------
    <keras.engine.sequential.Sequential object>
        Returns the CNN model object
    """

    # Define a sequential model and its layers
    cnn_model = models.Sequential([
        # Convolution layers
        # Conv layer 1
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=IMAGE_SHAPE, padding='same'),
        layers.MaxPooling2D((2, 2)),  # reduce image size

        # Conv layer 2
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=IMAGE_SHAPE, padding='same'),
        layers.MaxPooling2D((2, 2)),  # reduce image size

        # Conv layer 3
        layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=IMAGE_SHAPE, padding='same'),
        layers.MaxPooling2D((2, 2)),  # reduce image size

        # Conv layer 4
        layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', input_shape=IMAGE_SHAPE, padding='same'),
        layers.MaxPooling2D((2, 2)),  # reduce image size

        # Dense layers
        layers.Flatten(),  # Flatten before dense layers
        layers.Dense(200, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(50, activation='relu'),

        layers.Dense(10, activation='softmax')
    ])

    return cnn_model


def train_model(model, dataset):
    """
    Trains the inputted CNN model and fits it to both the train and test (validation) datasets

    Parameters
    ----------
    model : <keras.engine.sequential.Sequential object>
        CNN model object

    dataset : list<np.ndarray>
        A list containing x_train, x_test, y_train, y_test respectively

    Returns
    -------
    <keras.callbacks.History object>
        History object that retains the model training metrics
    """

    # Unpack the dataset from the given list
    x_train, x_test = dataset[0], dataset[1]
    y_train, y_test = dataset[2], dataset[3]

    # Compile the model with common adaptive optimizer
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Fit the CNN and use the test datasets as the validation data
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    # Evaluate the model (comes out to around 74.91%)
    print("\n\nFinal Evaluation on test set:")
    cnn.evaluate(x_test, y_test)

    return history


def plot_results(model_history):
    """
        Plots the accuracy and loss over Epochs to visualize model performance

    Parameters
    ----------
    model_history : <keras.callbacks.History object>
        History object that retains the model training metrics

    Returns
    -------
    None
    """

    # Plot the Accuracy Results
    plt.plot(model_history.history["accuracy"])
    plt.plot(model_history.history["val_accuracy"])
    plt.title("Model Performance Over Epochs (Accuracy)")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Epochs")
    plt.legend(["Train", "Val"])
    plt.show()

    # Now plot the Loss Results
    plt.plot(model_history.history["loss"])
    plt.plot(model_history.history["val_loss"])
    plt.title("Model Performance Over Epochs (Loss)")
    plt.ylabel("Loss")
    plt.xlabel("Number of Epochs")
    plt.legend(["Train", "Val"])
    plt.show()


if __name__ == '__main__':
    data = prepare_data()
    cnn = create_model()
    model_history = train_model(cnn, data)
    plot_results(model_history)
