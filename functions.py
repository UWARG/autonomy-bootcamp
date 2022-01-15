import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

def load_normalize_categorize_data():
    """loads in dataset and normalizes and categorizes the data to be fed into the CNN model

    Returns:
        numpy arrays for training and testing data
    """
    data = tf.keras.datasets.cifar10
    
    # Loading in the training and test data
    # x_train --> pixel values
    # y_train --> labels (airplane, cats, dog, etc.)
    (xTrain, yTrain), (xTest, yTest) = data.load_data()

    # Normalize the data to convert from values of (0 to 255) to just (0 and 1) for the CNN
    xTrain = tf.keras.utils.normalize(xTrain, axis=1)
    xTest = tf.keras.utils.normalize(xTest, axis=1)

    # Categorizing Data
    yTrain = tf.keras.utils.to_categorical(yTrain)
    yTest = tf.keras.utils.to_categorical(yTest)
    
    return xTrain, yTrain, xTest, yTest

def build_model(filters, kernelSize, activationConv2d, padding, inputShape, dropoutFactor, poolSize, units, activationDense):
    """generates a Sequential model with given parameters of filters, kernel sizes, and other variables of model

    Parameters
    ----------
        filters : list
            list containing number of filters in each convolution layer correspondent to index in list
        kernelSize : tuple (height, width)
            tuple containing size of kernel box
        activationConv2d : str
            string naming activation method of convolution and initial dense layers
        padding : str
            string naming padding type of model
        inputShape : tuple (height, width, depth)
            tuple containing shape of input data
        dropoutFactor : float
            float specifying the dropout factor of the dropout layer
        poolSize : tuple (height, width)
            tuple specifying pool box size
        units : list
            list containing number of units in each dense layer correspondent to index in list
        activationDense : str
            string containing activation method of final dense layer

    Returns
    -------
        keras.engine.sequential.Sequential object : Sequential Model
    """
    model = tf.keras.models.Sequential()
    
    # Concv2D Layers
    for i in range(len(filters)):
        if i == 0:
            model.add(tf.keras.layers.Conv2D(
                filters = filters[i],
                kernel_size = kernelSize,
                # activation method --> method to decide when and which nodes to fire
                activation = activationConv2d,
                # padding --> adding layers of zero around input images to have the output image be the same size as original input image
                padding = padding,
                # height, width, depth                
                input_shape = inputShape
            ))
        else:
            model.add(tf.keras.layers.Conv2D(filters = filters[i], kernel_size = kernelSize, activation = activationConv2d))
        model.add(tf.keras.layers.MaxPool2D(poolSize))
        model.add(tf.keras.layers.Dropout(dropoutFactor))        
        
    # Flattening Layers
    model.add(tf.keras.layers.Flatten())
    
    # Dense Layers
    for i in range(len(units)):
        if i == len(units)-1:
            model.add(tf.keras.layers.Dense(units = units[i], activation = activationDense))
        else:
            model.add(tf.keras.layers.Dense(units = units[i], activation = activationConv2d))
     
    # Seeing summary of model
    model.summary()
    
    return model

def plot_graph(x, y, fmt: str = None, label: str = None, title: str = None, xLabel: str = None, yLabel: str = None, legend: bool = False):
    """
    Returns an image of a plot graph through pyplot using given parameters

    Parameters
    ----------
        x : array or scalar
            variable containing array to be included on x axis of plot
        y : array or scalar
            variable containing array to be included on y axis of plot
        fmt : str, optional
            variable containing formatting for plot lines such as type of line (dotted, dashed, solid, etc.) and color of lines 
        label : str, optional
            label for legend
        title : str, optional
            title of plotted graph
        xLabel : str, optional
            label of x-axis
        yLabel : str, optional
            label of y-axis
        legend : bool, optional
        
    Returns
    -------
        image of plotted graph
    """
    plt.plot(x, y, f"{fmt}", label=f"{label}")
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    if legend:
        plt.legend()
    plt.show()