# tensorflow imports
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# misc. imports
import matplotlib.pyplot as plt

def load_data():
    '''
    Prepare CIFAR-10 training, validation and test data before input into network
    '''
    
    # load cifar10 dataset from keras
    (trainImages, trainLabels), (testImages, testLabels) = datasets.cifar10.load_data()
    
    # vectorize labels using one-hot encoding
    trainLabels = to_categorical(trainLabels)
    testLabels = to_categorical(testLabels)

    # normalize image datasets by scaling values to [0,1]
    trainImages = trainImages/255.0
    testImages = testImages/255.0

    # set size of validation set to one batch size defined in CIFAR 10 docs as 10,0000 images (around 16.6% all data)
    # rationale: common to set around 17-20% of all data (training + test) as a separate validation set
    VALID_SIZE = 7500

    # split training dataset into validation and training sets
    # rationale: test dataset is used as final evaluation to check true accuracy of model
    #            hence testing data should not be used as validation
    # note: only using simple hold out validation (k-fold cross validation is overkill)
    validImages, validLabels = trainImages[:VALID_SIZE], trainLabels[:VALID_SIZE]
    trainImages, trainLabels = trainImages[VALID_SIZE:], trainLabels[VALID_SIZE:]
    
    return (trainImages, trainLabels), (validImages, validLabels), (testImages, testLabels)

def create_model():
    '''
    Create model based on simple VGG CNN architecture with additional regularization layers
    sources: Deep Learning with Python Chapter 5
             https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
    '''

    # create sequentially layered model
    model = models.Sequential()

    # conv2D -> batch norm -> conv2D -> batch norm -> max pool -> dropout
    # note: conv2d padding set to 'same' to retain border information in feature maps & to not change tensor dimensions
    #       batch norm added after each conv2d layer to normalize outputs before input into next layer
    #       max pool added to reduce computing cost & improve model efficiency
    #       dropout added to further reduce capacity & overfitting
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.2))

    # same block as above with following changes:
    #   increase filter size to 64 for more feature extraction
    #   compensate increased overfitting due to increased filter size by increasing dropout to 0.3
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.3))

    # same block as above with following changes:
    #   increase filter size to 128 for more feature extraction
    #   compensate increased overfitting due to increased filter size by increasing dropout to 0.4
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.4))

    # flatten -> relu activation -> batch norm -> dropout -> softmax activation
    # final classifier to determine labels probabilities
    # note: flatten required as data preprocessing for dense layers
    #       batch norm and dropout applied before final softmax layer as final regularization layers
    #       softmax activation used since this is a multi label classification problem
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    # print summary of model
    model.summary()

    return model


def plot_results(history):
    '''
    Create plots of training and validation loss functions and accuracies
    '''

    # create and title plot w/ two subplots
    fig, (loss, acc) = plt.subplots(2)
    fig.suptitle('Loss and Accuracy Plots')

    # plot training & validation loss functions per epoch
    loss.plot(history.history['loss'], color='blue', label='training loss')
    loss.plot(history.history['val_loss'], color='red', label='validation loss')
    loss.set_xlabel('Epoch')
    loss.set_ylabel('Loss')
    loss.legend(loc='upper right')

    # plot training & validation accuracy per epoch
    acc.plot(history.history['accuracy'], color='blue', label='training accuracy')
    acc.plot(history.history['val_accuracy'], color='red', label='validation accuracy')
    acc.set_xlabel('Epoch')
    acc.set_ylabel('Accuracy')
    acc.legend(loc='lower right')

if __name__ == '__main__':

    # prepare and load cifar10 data
    (trainImages, trainLabels), (validImages, validLabels), (testImages, testLabels) = load_data()

    # define network architecture
    model = create_model()

    # compile model
    # note: rmsprop used as per recommendation in 'Deep Learning with Python' Chapter 4
    #       adam or other optimizers are also valid, but trying SGD with customized parameters proved less efficient
    #       (i.e. i found it difficult to determine an appropriate learning rate & momentum)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # determine batch size to be used in model training
    BATCH_SIZE=128

    # perform data augmentation to produce more training data
    # note: CIFAR-10 images are low pixel, and thus image transformations should be used very sparingly
    #       training on too different looking images will produce inaccurate predictions
    #       here, only width & height shift ranges and horizontal flips are changed
    dataGen = ImageDataGenerator(width_shift_range=0.1,
                                height_shift_range=0.1,
                                horizontal_flip=True)

    # create iterator using training data
    trainIterator = dataGen.flow(trainImages, trainLabels, batch_size=BATCH_SIZE)

    # calculate step size used per epoch
    # note: found it's common to set the steps per epoch as train length / batch size
    steps = int(trainImages.shape[0]/BATCH_SIZE)

    # train model
    # note: set epoch size to 100 (to achieve 80%+ accuracies)
    #       validation data does not use testing data, uses separate validation data as processed in load_data()
    history = model.fit(trainIterator, steps_per_epoch=steps, batch_size=BATCH_SIZE, epochs=100, validation_data=(validImages, validLabels))

    # plot the training and validation results
    plot_results(history)

    # evaluate model on test data (previously unseen) and print results
    testLoss, testAcc = model.evaluate(testImages, testLabels)
    print('\nTEST LOSS RESULT: ' + str(testLoss))
    print('TEST ACC RESULT: ' + str(testAcc))