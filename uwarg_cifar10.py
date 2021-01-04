# loading machine learning library
import tensorflow as tf

import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# loading data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

"""
# Plotting image data
"""

# dataset labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

print('train_images shape:', train_images.shape)
print('test_images shape:', test_images.shape)

print('train_labels shape:', train_labels.shape)
print('test_labels shape:', test_labels.shape)

"""
# Prepping Data
"""

# normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# one hot encoding into binary matrices
# each vector represents an indivdual class
train_labels = keras.utils.to_categorical(train_labels, len(class_names))
test_labels = keras.utils.to_categorical(test_labels, len(class_names))

print('train_labels shape:', train_labels.shape)
print('test_labels shape:', test_labels.shape)

"""
# Creating CNN
"""

# defining CNN model
model = models.Sequential()

# adding multiple convolutional layers
# neural network architecture from https://www.kaggle.com/msameerkhan/cifar10-with-cnn

# CONV2D => CONV2D => BATCHNORMALIZATION => POOL => DROPOUT
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

# CONV2D => CONV2D => BATCHNORMALIZATION => POOL => DROPOUT
model.add(layers.Conv2D(64, (3, 3), padding='same',activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

# CONV2D => CONV2D => BATCHNORMALIZATION => POOL => DROPOUT
model.add(layers.Conv2D(128, (3, 3), padding='same',activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

# FLATTERN => DENSE => RELU => DROPOUT
model.add(layers.Flatten())

model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.5))

# a softmax classifier
model.add(layers.Dense(len(class_names),activation='softmax'))

model.summary()

"""
# Training
"""

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# preprocessing and data augmentation
# image augmentation based on https://www.kaggle.com/msameerkhan/cifar10-with-cnn
datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.,
        zoom_range=0.,
        fill_mode='nearest',
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        validation_split=0.0)
datagen.fit(train_images)

history = model.fit(datagen.flow
                    (train_images,
                     train_labels,
                     batch_size=64),
                    epochs=40,
                    validation_data=(test_images, test_labels),
                    workers=4)

"""
# Model evaluation
"""

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
