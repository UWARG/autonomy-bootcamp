# @author: Ishman Mann
# @date: 13/10/2022
# 
# @description:
#   Classification model for CIFAR-10 dataset using a CNN in TensorFlow
#
# @references:
#   https://www.youtube.com/watch?v=tPYj3fFJGjk&t=3961s&ab_channel=freeCodeCamp.org
#   https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions
#   https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
#   https://www.tensorflow.org/tutorials/images/data_augmentation

#------------------------------------------------------------------------------------------------------------------------------------------------------------

# Imports

import matplotlib.pyplot as plt
import numpy as np

from sklearn import model_selection

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

from keras import datasets, layers, models

#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Loading the dataset

# as_supervised=True yeilds (x, y) tuples instead of a dictionary
(datasetTrain, datasetValidate, datasetTest), datasetInfo = tfds.load('cifar10', split=['train[:80%]','train[80%:]', 'test'], 
                                                     shuffle_files=True, as_supervised=True, with_info=True)

NUM_CLASSES = datasetInfo.features['label'].num_classes
CLASS_NAMES = datasetInfo.features['label'].names

#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Data preprocessing and augmentation


IMG_SIZE = 32
dataResizingRescaling = keras.Sequential([layers.Resizing(IMG_SIZE, IMG_SIZE), # for sanity, ensure images are same size 
                                         layers.Rescaling(1.0/255)]) 

dataAugmention = keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomTranslation(0.2, 0.2)
  #layers.RandomZoom(width_factor=(-0.2,0.2), height_factor=((-0.2,0.2)))

  # **need to add more transformations here
  # **consider adding custom transformations
  # **reference https://www.tensorflow.org/tutorials/images/data_augmentation, https://www.tensorflow.org/api_docs/python/tf/keras/layers
])


def prepare_data(dataset, batchSize=32, training=False, numAugmentations=0, shuffleBufferSize=1000):

  # rescale and resize all datasets
  dataset = dataset.map(lambda x, y: (dataResizingRescaling(x) , y),
                        num_parallel_calls=tf.data.AUTOTUNE) # AUTOTUNE means dynamically tuned based on available CPU
  
  if (training):

    # desired numAugmentations are made and concatenated to original dataset
    if (numAugmentations > 0):
      augmentation = dataset.repeat(count=numAugmentations)
      augmentation = augmentation.map(lambda x, y: (dataAugmention(x, training=True), y),
                                      num_parallel_calls=tf.data.AUTOTUNE)
      dataset = dataset.concatenate(augmentation)

    # shuffle and batch dataset
    dataset = dataset.shuffle(buffer_size=shuffleBufferSize)
    dataset = dataset.batch(batchSize)
  
  else:

    # batch dataset
    dataset = dataset.batch(batchSize)

  # prefetch to overlap dataset preprocessing and model excecution -> faster run time
  # (not using dataset.cache().prefetch, incase dataset is too large for cache storage) 
  return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


# Will run augmentation seperate to model creation for efficiency
datasetTrain = prepare_data(datasetTrain, training=True, numAugmentations=0)
datasetValidate = prepare_data(datasetValidate)
datasetTest = prepare_data(datasetTest)

#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Creating the model

model = keras.Sequential([
    
    # Convolutional base
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='valid', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='valid', activation='relu'),
    layers.MaxPooling2D(),
    
    # Dense Layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')

])

model.summary()

model.compile(optimizer='adam',
              #loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), # can't use softmax in last layer when from_logits=True
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), # from_logits=False -> data is a probability distribution 
              metrics=['accuracy']) #note to self: research different types of Keras metrics


#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Training the model

NUM_EPOCHS = 10
history = model.fit(datasetTrain,
                    validation_data=datasetValidate,
                    epochs=NUM_EPOCHS
                    ) 

#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Testing the model

loss, accuracy = model.evaluate(datasetTest)
print(loss, accuracy)