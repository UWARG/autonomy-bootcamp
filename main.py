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

from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Your working code here

# This code was initially written on Google Collab, hence the packing
# of everything into functions rather than a single script.

checkpoint_path = "{model_name}.ckpt"

def load_dataset():
    (trainX, trainY), (testX, testY) = cifar10.load_data()

    trainY, testY = to_categorical(trainY), to_categorical(testY)

    return trainX, trainY, testX, testY


def normalize_pixels(train, test):
    # Normalizes the input to have a range between 0 and 1
    train, test = train.astype('float32'), test.astype('float32')
    train /= 255
    test /= 255
    return train, test


def diagnostics(history, model_name):
    # Plots cross entropy and accuracy based off training and validation, to determine
    # how well the model is performing (over-fitting vs under-fitting)
    pyplot.subplot(1, 2, 1)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')

    pyplot.subplot(1, 2, 2)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')

    pyplot.savefig(model_name + '_plot.png')
    pyplot.close()


def evaluate_saved_model(model, model_name):
    # Loads a saved model and evaluates it for its accuracy
    trainX, trainY, testX, testY = load_dataset()

    trainX, testX = normalize_pixels(trainX, testX)

    model.load_weights(checkpoint_path.format(model_name=model_name))
    _, acc = model.evaluate(testX, testY)
    print('Saved model, accuracy: {:5.2f}%'.format(100 * acc))


def run_test_harness_with_augments(model, model_name):
    trainX, trainY, testX, testY = load_dataset()

    trainX, testX = normalize_pixels(trainX, testX)

    # Creates a generator that modifies the image according to the given parameters;
    # in this case, modifies the width and height by 10% each, and flips the image horizontally
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    it_train = datagen.flow(trainX, trainY, batch_size=64)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path.format(model_name=model_name),
                                                     save_weights_only=True,
                                                     monitor='val_accuracy',
                                                     mode='max',
                                                     save_best_only=True,
                                                     verbose=1)

    # Specify the steps per epoch, since the ImageDataGenerator produces images indefinitely
    steps = int(trainX.shape[0] / 64)
    # Actually train
    history = model.fit(x=it_train, steps_per_epoch=steps, epochs=200, validation_data=(testX, testY),
                        callbacks=[cp_callback])
    _, acc = model.evaluate(testX, testY)
    print('> %.3f' % (acc * 100.0))
    diagnostics(history, model_name)


def vgg_dropout_block():
    model = Sequential()
    model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(64, 3, activation='relu', padding='same'))
    model.add(Conv2D(64, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


run_test_harness_with_augments(vgg_dropout_block(), 'vgg_dropout_block_augmented')