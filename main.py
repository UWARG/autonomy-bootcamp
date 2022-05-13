import ssl

import matplotlib.pyplot as plt
import tensorflow as tf

ssl._create_default_https_context = ssl._create_unverified_context


def load_data():

    """
    This function loads the data onto the training and testing variables

    Furthermore, it one hot encodes the y test and train variables and 
    normalizes the x train and test variables to be between 0 and 1

    Returns
    -------
    Returns a tuple of training and testing values for x and y
    """

    # Loading data to develop the model
    (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()

    # Normalizing data inputs to be between 0 and 1
    xTrain = xTrain.astype("float32")
    xTest = xTest.astype("float32")
    xTrain = xTrain/255.0
    xTest = xTest/255.0

    # One hot encoding target values
    yTrain = tf.keras.utils.to_categorical(yTrain)
    yTest = tf.keras.utils.to_categorical(yTest)

    return xTrain, xTest, yTrain, yTest


def create_model(xTrain, yTest):

    """
    This funtion creates the sequential model for classifying the CIFAR-10 
    dataset by adding multiple filters and connecting each neuron

    It follows the order of convolutional layer, followed by activation or
    pooling layer, and finally a dropout and normalization layer

    Flattens and compiler model before returning

    Requires
    -------
    X_train and y_test data

    Returns
    -------
    Returns the sequential model that contains all added layers
    """

    # Creating the model
    classificationModel = tf.keras.Sequential()

    # Setting up the channels and filters (We use 3x3 filters)
    classificationModel.add(
        tf.keras.layers.Conv2D(
            32, (3, 3), 
            input_shape=xTrain.shape[1:], 
            padding="same"
        )
    )
    classificationModel.add(tf.keras.layers.Activation("relu"))
    classificationModel.add(
        tf.keras.layers.Conv2D(
            32, 3, 
            input_shape=(32, 32, 3), 
            activation="relu", 
            padding="same"
        )
    )
    
    # Adding a dropout layer to prevent overfitting
    classificationModel.add(tf.keras.layers.Dropout(0.2))

    # Normalizing data going to the next layer
    classificationModel.add(tf.keras.layers.BatchNormalization())

    # Next convolution layer
    classificationModel.add(
        tf.keras.layers.Conv2D(
            64, 3, 
            activation="relu", 
            padding="same"
        )
    )

    # Pooling layer for the next layer
    classificationModel.add(tf.keras.layers.MaxPooling2D(2))

    # Dropout layer for the next layer
    classificationModel.add(tf.keras.layers.Dropout(0.2))

    # Normalizing data going to even further layers
    classificationModel.add(tf.keras.layers.BatchNormalization())

    # Final layers
    classificationModel.add(
        tf.keras.layers.Conv2D(
            128, 3, 
            activation="relu", 
            padding="same"
        )
    )
    classificationModel.add(tf.keras.layers.Dropout(0.2))
    classificationModel.add(tf.keras.layers.BatchNormalization())

    # Flattening the model and adding another dropout layer
    classificationModel.add(tf.keras.layers.Flatten())
    classificationModel.add(tf.keras.layers.Dropout(0.2))    

    # Creating a 32 layer neural network
    classificationModel.add(tf.keras.layers.Dense(32, activation="relu"))
    classificationModel.add(tf.keras.layers.Dropout(0.3))
    classificationModel.add(tf.keras.layers.BatchNormalization())

    # Selecting the neuron with the highest probability for classifying
    classificationModel.add(
        tf.keras.layers.Dense(
            yTest.shape[1], 
            activation="softmax"
        )
    )

    # Compiling the model
    classificationModel.compile(
        loss="categorical_crossentropy", 
        optimizer="adam", 
        metrics=["accuracy"]
    )

    return classificationModel


def generate_plot(history):

    """
    Function to generate the plot that displays both the loss
    as well as the accuracy of the model

    Requires
    -------
    history; the fitted TensorFlow model variable
    """

    # Finding the loss in the plot
    plt.subplot(211)
    plt.title("Loss")
    plt.plot(history.history["loss"], color="blue", label="train")
    plt.plot(history.history["val_loss"], color="orange", label="test")

    # Finding the accuracy of the plot
    plt.subplot(212)
    plt.title("Accuracy")
    plt.plot(history.history["accuracy"], color="blue", label="train")
    plt.plot(history.history["val_accuracy"], color="orange", label="test")

    # Display the plots
    plt.savefig("plot.png")
    plt.close()

def run_model():
    
    """
    Main function that calls other functions to run the model

    Trains the model against the dataset and classifies images 
    according to the CIFAR-10 dataset

    Furthermore, generates plot and shows the accuracy of the
    model
    """

    xTrain, xTest, yTrain, yTest = load_data()
    model = create_model(xTrain, yTest)

    # Training the model and storing its efficiency
    history = model.fit(
        xTrain, yTrain, 
        validation_data=(xTest, yTest), 
        epochs=25, batch_size=64
    )
    efficiency = model.evaluate(xTest, yTest, verbose=0)

    print("Accuracy: %.2f%%" % (efficiency[1]*100))

    # Plots the graphs
    generate_plot(history)


# Run the classification model
run_model()
