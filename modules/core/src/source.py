'''
    File: source.py
    Author: WARG
    Section: LICENSE
    
    Copyright (c) 2015-2016, Waterloo Aerial Robotics Group (WARG)
    All rights reserved.

    This software is licensed under a modified version of the BSD 3 clause license
    that should have been included with this software in a file called COPYING.txt
    Otherwise it is available at:
    https://raw.githubusercontent.com/UWARG/computer-vision/master/COPYING.txt
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import os
import sys
import pickle

IMAGES_PER_BATCH = 10000
NUM_BATCHES = 5
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CATEGORIES = 10

def readFile(cwd, file, encoding):
    ''' Reads the file located in the directory cwd/file with the encoding given '''
    # Read the file at the given path
    with open(os.path.join(cwd, file), 'rb') as fo:
        inputStream = pickle.load(fo, encoding=encoding)
    
    #Return an input stream
    return inputStream

def parse_data(directory):
    ''' Extracts data from the CIFAR-10 dataset files from a given directory. '''
    # Find the CIFAR dataset and metadata file from the provided directory
    directory = os.path.join(directory, 'modules', 'core', 'src')
    metadata = readFile(directory, 'batches.meta', 'utf-8')

    # Create an empty numpy array to hold 5 batches of 10 000 colour images of size 32x32
    xTrainBatch = np.zeros((NUM_BATCHES, IMAGES_PER_BATCH, IMG_HEIGHT, IMG_WIDTH, 3), np.uint8())
    # Create an empty numpy array to hold labels associated with the images
    yTrainBatch = np.zeros((NUM_BATCHES, IMAGES_PER_BATCH), np.uint8())

    # Open each of the training files
    TRAINING_FILES = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    for batchNum, file in enumerate(TRAINING_FILES):
        # Call a function to read in the data: received format is a dictionary containing 'labels' and 'data' (the images)
        batchData = readFile(directory, file, 'bytes')

        # Data arrives as a 10000x3072 numpy array, where each row contains 1024 red, then blue, then green pixels
        # corresponding to the 32x32 grid. Split the 3072 element 1D numpy array into 3 channels, containing a 32x32 array,
        # then flip the array so that the format is (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
        xTrainBatch[batchNum] = np.moveaxis(np.reshape(batchData[b'data'], (IMAGES_PER_BATCH, 3, IMG_HEIGHT, IMG_WIDTH)), 1, -1)
        # Get a list of labels associated with the training data
        yTrainBatch[batchNum] = np.array(batchData[b'labels'])

    # Do the same for the test files
    batchData = readFile(directory, 'test_batch', 'bytes')
    xTest = (np.moveaxis(np.reshape(batchData[b'data'], (10000, 3, 32, 32)), 1, -1))
    yTest = np.array(batchData[b'labels'], np.uint8())

    # Return a tuple with the training data
    return (xTrainBatch, xTest, yTrainBatch, yTest)

def generate_model():
    ''' Builds and compiles a convolutional neural network to solve the CIFAR-10 classification '''
    model = tf.keras.Sequential(
        [
            # Input layer has shape (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
            tf.keras.layers.Conv2D(8, (5, 5), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(16, (5, 5), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),

            # Output layer has size 10: one neuron for each output
            tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
        ]
    )
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model
