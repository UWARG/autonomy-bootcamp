####################################################################################################
# CIFAR-10 Image CLassifier - Ayan Hafeez                                                          #
# Created for the UWARG-CV Bootcamp                                                                #
# Plots are present in the repo as well as a schematic of the standard model being used to train   #
#                                                                                                  #
####################################################################################################


import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow

from tensorflow.keras import Input as inputNode
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import (
    Conv2D as conv_2d,
    MaxPooling2D as max_pooling2d,
    Dropout as dropout,
    Dense as dense,
    Flatten as flatten,
)
from tensorflow.keras.models import Model as model_f
from tensorflow.keras.preprocessing.image import ImageDataGenerator as data_gen
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

import constants



def pull_dataset():
    """
    Pulls Cifar-10 dataset that is integrated into Keras. 

    Returns CIFAR-10 dataset.
    """
    return tensorflow.keras.datasets.cifar10.load_data()



def preprocess_data(data):
    """
    Preprocesses the data in the CIFAR-10 dataset; first rescales both the training and testing data 
    from 0-->255 to 0-->1 in order to train our model, and also splits 20% of the training batch 
    into a validation batch to be used later. All data are returned as generators. Batch size is also
    specified here.
    
    Returns tuple of generators for each batch of data. 
    """
    ((x_train, y_train), (x_test, y_test)) = data
    preprocessor = data_gen(rescale = constants.RESCALE_FACTOR, validation_split = constants.VALIDATION_SPLIT)
    return (
        preprocessor.flow(x = x_train, y = y_train, batch_size = constants.BATCH_SIZE, subset = "training"),
        preprocessor.flow(x = x_test, y = y_test, batch_size = constants.BATCH_SIZE),
        preprocessor.flow(x = x_train, y = y_train, batch_size = constants.BATCH_SIZE, subset = "validation"),
        )


def mod_vgg_layer(input, hp, convNum, filterNum):
    """
    Creates a modified version of the VGG block, takes parameters for number of convolution layers 
    and corrresponding filters. The VGG Block is structured with a 2d convolution layer to create 
    feature maps, and then followed up by a max pooling layer to down-sample, reducing width/height 
    of the payload. There is a final dropout layer which mitigates overfitting. 

    Returns mutated payload.
    """
    for i in range(convNum):
        # creates convNum x convolution layers
        input = conv_2d(filterNum, (3, 3), padding = "same", activation = "relu")(input) 
        # max pooling layer
    input = max_pooling2d((2, 2), strides=(1, 1))(input)
    # dropout layer
    input = dropout(constants.DROPOUT_FACTOR)(input)  
    return input



def create_model(hp):
    """
    Creates a model using the Keras Functional API. Layers three modded VGG Blocks with abritary
    conv2d layer counts, followed by a conversion to a 1 dimensional tensor which is used to create
    a fully-connected layer composed of a 128 unit dense layer, dropout, and final layer with 
    softmax activation to get probability distribution for each class. Softmax is used since this
    is a multi-class classification problem. 

    Returns compiled model. 
    """
    img_inputs = inputNode(shape = constants.IMG_SIZE)
    # block 1
    layer = mod_vgg_layer(img_inputs, hp, 3, constants.VGG_LAYER1_FILTER) 
    # block 2
    layer = mod_vgg_layer(layer, hp, 2, constants.VGG_LAYER1_FILTER) 
    # block 3
    layer = mod_vgg_layer(layer, hp, 2, constants.VGG_LAYER2_FILTER) 
    # flattening the layer
    layer = flatten()(layer) 
    layer = dense(constants.FINAL_DENSE_UNITS, activation = "relu")(layer) 
    layer = dropout(constants.DROPOUT_FACTOR)(layer)
    # final layer with softmax
    layer = dense(constants.CLASS_COUNT, activation = "softmax")(layer) 
    model = model_f(inputs = img_inputs, outputs = layer)
    # printing the model summary to a .txt
    with open("model_layout.txt", "w") as file:  
        model.summary(print_fn = lambda x: file.write(x + "\n"))
    # actually compiling the model while specifying the optimizer and loss function
    model.compile( 
        optimizer = "sgd",
        # we use sparse_categorical_crossentropy since  our data has labels in an integer vector
        loss = "sparse_categorical_crossentropy", 
        metrics = ["sparse_categorical_accuracy"],
    )
    return model


def tune_model(traindata, validata):
    """
    Tunes models created by create_model using KerasTuner. Using a random search, tests predefined
    hyperparameters in the model to optimize for accuracy. 

    Returns HyperParameter object containing the best parameters for the current architecture. 
    """
    tuner = kt.RandomSearch(
        hypermodel = create_model,
        max_trials = 5,
        objective = "val_sparse_categorical_accuracy",
        overwrite = True,
    )
    tuner.search(traindata, epochs = 3, validation_data = validata)

    return tuner.get_best_hyperparameters()[0]


def plot_data(data):
    """
    Plots accuracy and loss plots for the model training.
    """
    plt.title("Accuracy vs Epoch")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.plot(data.history["sparse_categorical_accuracy"])
    plt.plot(data.history["val_sparse_categorical_accuracy"])
    plt.legend(["training", "validation"], loc = "lower right")
    plt.savefig(fname = "accuracy_plot")
    plt.show()

    plt.title("Loss vs Epoch")
    plt.ylabel("loss")
    plt.xlabel("Epoch")
    plt.plot(data.history["loss"])
    plt.plot(data.history["val_loss"])
    plt.legend(["training", "validation"], loc = "upper right")
    plt.savefig(fname = "loss_plot")
    plt.show()


def choose_mode():
    """
    Prompts user to either enter tuning mode, or to train the model.

    Returns boolean corresponding to choice.
    """
    answer = input("Would you like to tune the model hyperparameters, or go ahead and train? (1- tune, 2- train) \n")
    if answer == "1":
        return True
    return False


def choose_cont():
    """
    Prompts user to continue with training after tuning is complete. 

    Returns 0 if user chooses to continue, otherwise exits the program. 
    """
    answer = input("Model Tuning Complete. Would you like to continue with training? (y / n) \n")
    if answer == "y":
        return 0
    else:
        sys.exit()



def main():
    """
    Driver function to run the image classifier. 
    """
    # pulls and preprocesses the data
    (train_iter, test_iter, vali_iter) = preprocess_data(pull_dataset())
    # initializes model with KerasTune hyperparameter container
    model = create_model(kt.HyperParameters())
    # prompts user to choose whether to tune or to train directly using preset values
    if choose_mode():
        # tunes model and extracts the best parameters 
        best_params = tune_model(train_iter, vali_iter)
        # creates a model from the best parameters 
        model = create_model(best_params)
        # prompts the user whether to continue or not
        choose_cont()
    # trains the model on the preprocessed data
    data = model.fit(
        train_iter,
        validation_data = vali_iter,
        validation_steps = len(vali_iter),
        steps_per_epoch = len(train_iter),
        epochs=constants.EPOCH_NUM,
    )
    # plots the loss and accuracy plots
    plot_data(data)
    # evaluates the model's final loss and accuracy on the test dataset and outputs the values
    (testLoss, testAccuracy) = model.evaluate(test_iter)
    print("Final Model Loss = ", testLoss)
    print("Final Model Accuracy = ", testAccuracy)


if __name__ == "__main__":
    # runs driver code
    main()
