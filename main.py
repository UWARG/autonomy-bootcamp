'''
    File: main.py
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
import os
import sys
import matplotlib.pyplot as plt


from modules.core.src.source import parse_data, generate_model

EPOCHS = 10


def main():
    ''' Parses image data, trains model, and displays plots of training '''
    # Get a parsed version of the batch data
    # Note: xTrainBatch and yTrainBatch contain 5 batches of data in a numpy array of size (5x10x32x32x3)
    xTrainBatch, xTest, yTrainBatch, yTest = parse_data(os.getcwd())

    # Get a compiled neural network compatible with the batch data
    model = generate_model()

    # Print a summary of the neural network topology to the console
    model.summary()

    #Variables to hold loss values over epochs
    trainLoss = list()
    testLoss = list()

    # Train the model with each batch of data available
    for batch in range(0,5):
        print("\nBatch #" + str(batch+1) + "\n___________________________")
        history = (model.fit(xTrainBatch[batch], yTrainBatch[batch], epochs=EPOCHS, validation_data=(xTest, yTest)))
        trainLoss += list(history.history['loss'])
        testLoss += list(history.history['val_loss'])

    # Count the number of epochs
    epochCount = range(1, len(trainLoss) + 1)

    # Plot a distribution of training and testing loss over epochs
    plt.plot(epochCount, trainLoss, 'r--')
    plt.plot(epochCount, testLoss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    ''' Calls the main function for the program '''
    main()
