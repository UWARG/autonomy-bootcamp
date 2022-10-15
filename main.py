# @author: Ishman Mann
# @date: 13/10/2022
# 
# @description:
#   Classification model for CIFAR-10 dataset using a CNN in TensorFlow
#
# @references:
#   https://www.youtube.com/watch?v=tPYj3fFJGjk&t=3961s&ab_channel=freeCodeCamp.org
#   https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions

# --------------------------------------------------------------------------------------------------------

# Imports

import matplotlib.pyplot as plt
import numpy as np

from sklearn import model_selection

import tensorflow as tf
from tensorflow import keras
import tensorflow datasets as tfds

from keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator # baddd, keras.preprocessing is deprecated


# Loading the dataset

(trainImages, trainLabels), (testImages, testLabels) = keras.datasets.cifar10.load_data()

# Data preprocessing and augmentation

trainImages, testImages = trainImages/255, testImages/255 

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(trainX, validationX, trainY, validationY) = model_selection.train_test_split(trainImages, trainLabels, 
                                                                              test_size = 0.20, 
                                                                              random_state = 42)



# badddd, keras.preprocessing is deprecated
trainAugmentedDatagen = ImageDataGenerator(rotation_range=20,
                                           zoom_range=0.15,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.15,
                                           horizontal_flip=True,
                                           fill_mode="nearest")

trainAugmented = trainAugmentedDatagen.flow(trainX, trainY, batch_size=32)


#   consider having a train, validation, and test set -> ask: is validation really necessary
#   use the img transformation thing
#   think about efficiency when using tf.keras.Model.fit function
#      should I store the images first??


