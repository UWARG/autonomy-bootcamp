####################################################################################################
# CIFAR-10 Image CLassifier - Ayan Hafeez                                                          #
# Created for the UWARG-CV Bootcamp                                                                #
# Plots are present in the repo as well as a schematic of the standard model being used to train   #
#                                                                                                  #
####################################################################################################

import constants
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np  
import sys
import tensorflow

from tensorflow.keras import Input as input_node
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Conv2D as conv_2d, MaxPooling2D as max_pooling2d, Dropout as dropout, Dense as dense, Flatten as flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator as data_gen
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# pull the cifar10 dataset since it is a module in keras
def pull_dataset(): 
  return tensorflow.keras.datasets.cifar10.load_data() 
# preprocess images for insertion into the network
def preprocess_data(data): 
    (x_train, y_train), (x_test, y_test) = data
    preprocessor = data_gen(rescale=1.0/255.0, validation_split=0.2)
    return (preprocessor.flow(x=x_train, y=y_train, batch_size=constants.BATCH_SIZE, subset="training"),   
            preprocessor.flow(x=x_test, y=y_test, batch_size=constants.BATCH_SIZE), 
            preprocessor.flow(x=x_train, y=y_train, batch_size=constants.BATCH_SIZE, subset="validation"))

def mod_vgg_layer(input, hp, c, f):
    for i in range(c):
        input = conv_2d(f, (3,3), padding='same', activation="relu")(input)
    input = max_pooling2d((2,2), strides=(1,1))(input)
    return input
# create the CNN model (initializing layers) 
def create_model(hp):
   img_inputs = input_node(shape=constants.IMG_SIZE)
   layer = mod_vgg_layer(img_inputs, hp, 3, 64)
   layer = dropout(0.4)(layer)
   layer = mod_vgg_layer(layer, hp, 2, 64)
   layer = dropout(0.4)(layer)
   layer = mod_vgg_layer(layer, hp, 2, 128)
   layer = flatten()(layer) 
   layer = dense(128, activation="relu")(layer)
   layer = dropout(0.4)(layer)
   layer = dense(constants.CLASS_COUNT, activation="softmax")(layer)
   model = Model(inputs=img_inputs, outputs=layer)
   with open('model_layout.txt', 'w') as file: # modularize
       model.summary(print_fn=lambda x: file.write(x + '\n'))
   model.compile(optimizer='sgd', loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])   
   return model

def tune_model(traindata, validata):
    tuner = kt.RandomSearch(
        hypermodel=create_model,
        max_trials=5,
        objective="val_sparse_categorical_accuracy",
        overwrite=True,
    )
    tuner.search(traindata, epochs=3, validation_data=validata)
    
    return (tuner.get_best_hyperparameters()[0])

def plot_data(data):
    plt.title("Accuracy vs Epoch")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.plot(data.history["sparse_categorical_accuracy"])
    plt.plot(data.history["val_sparse_categorical_accuracy"])
    plt.legend(['training', 'validation'], loc='lower right')
    plt.savefig(fname="accuracy_plot")
    plt.show()

    plt.title("Loss vs Epoch")
    plt.ylabel("loss")
    plt.xlabel("Epoch")
    plt.plot(data.history["loss"])
    plt.plot(data.history["val_loss"])
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig(fname="loss_plot")
    plt.show()

def wait_for_user1():
    answer = input("Would you like to tune the model hyperparameters, or go ahead and train? (1- tune, 2- train) \n")
    if answer == "1":
        return True
    return False

def wait_for_user2():
    answer = input("Model Tuning Complete. Would you like to continue with training? (y / n) \n")
    if answer == "y":
        return 0
    else:
        sys.exit()
# driver function for classifier
def main():
    (train_iter, test_iter, vali_iter) = preprocess_data(pull_dataset())
    model = create_model(kt.HyperParameters())
    if(wait_for_user1()):
        best_params = tune_model(train_iter, vali_iter)
        model = create_model(best_params)
        wait_for_user2()
    data = model.fit(train_iter, validation_data=vali_iter, validation_steps=len(vali_iter), steps_per_epoch=len(train_iter), epochs=20)
    plot_data(data)
    (testLoss, testAccuracy)  = model.evaluate(test_iter)
    print("Final Model Loss = " , testLoss)
    print("Final Model Accuracy = " , testAccuracy)
  


if __name__ == "__main__":
    main()