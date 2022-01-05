import numpy as np  
import tensorflow as tf 
import matplotlib.pyplot as plot

# pull the cifar10 dataset since it is a module in keras
def pull_dataset(): 
  return tf.keras.datasets.cifar10.load_data()

# preprocess images for insertion into the network
def preprocess_data(): 
    return 0 

# create the CNN model (initializing layers) 
def create_model():
    return 0
# return array of layers to be sequentially used (remains to be seen whether this needs a whole func, probably not)
def layer_array(): 
    return 0
def main():
    (x_train, y_train), (x_test, y_test) = pull_dataset()


if __name__ == "__main__":
    main()