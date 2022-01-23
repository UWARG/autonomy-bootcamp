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

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


def process_data():
    """
    Loads CIFAR-10 dataset and returns the training and test set
    
    Parameters
    ----------
    None
        
    Returns
    -------
    (X_train, Y_train), (X_test, Y_test) : Tuple
        Returns CIFAR-10 normalized training and test data
    
    """
    
    #load dataset from keras
    (X_train, Y_train), (X_test, Y_test) = datasets.cifar10.load_data()
    
    #Normalize data, divide each pixel value by 255, since values range from 0-255 for RBG
    X_train = X_train / 255
    X_test = X_test / 255
    
    
    return (X_train, Y_train), (X_test, Y_test)
    
def get_model():
    """
    Creates a convolutional neural network with the help of a sequential model
    
    Parameters
    ----------
    None
    
    Returns
    -------
    cnn_model : keras.engine.sequential.Sequential
        Returns a CNN Model
    
    """
    
    cnn_model = models.Sequential([

        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    return cnn_model
def show_plots(history):
    """
    Displays a  plot of both training and validation losses over epochs
    
    Parameters
    ----------
    history : keras.callbacks.History
        The history object (recorded events) returned by fitting a model. Metrics are stored in a dictionary in the history member of the object returned.
        
    Returns
    -------
    None
    """
    
    # loss: value of loss function for your training data

    # see what categories we're able to plot
    print(history.history.keys()) 

    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])

    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    
    plt.show()
    
def main():
    """
    Main function processes, trains, and plots the metrics
    """
    
    (X_train, Y_train), (X_test, Y_test) = process_data()
    
    cnn_model = get_model()
    
    #fit and compile model
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_history = cnn_model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test)) 
    
    show_plots(cnn_history)
    
    
main()