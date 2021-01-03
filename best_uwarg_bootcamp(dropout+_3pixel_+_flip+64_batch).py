# -*- coding: utf-8 -*-
"""Best_UWARG_Bootcamp(dropout+ 3pixel + flip+64 batch)

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1E_WuH0zSDbQ9jioT5MnbGCay2rEaQVvH
"""

from keras.utils import to_categorical
from keras.datasets import cifar10
# load data set
(Train_X, Train_Y), (Test_X, Test_Y) = cifar10.load_data()

# convert categorical data to numeric using one-hot encoding
Train_Y = to_categorical(Train_Y)
Test_Y = to_categorical(Test_Y)

# set pixels to a value between 0 and 1
Train_X = Train_X / 255.0
Test_X = Test_X / 255.0

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization

# model using 3 blocks of VGG followed by fully connected layers
# using Batch Normalization and dropout to reduce overfitting
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.15))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.35))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.45))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
# perfoming slight bit of image augmentation
# there's only 32 by 32 pixels, I tried shifting by 4 and it was worse
# 3 seems to be the best option
datagen = ImageDataGenerator(
    width_shift_range=3, height_shift_range=3, horizontal_flip=True,
) # horrizontal and verticle shift by 3 pixels

from keras.callbacks import EarlyStopping
# stops the training early if improvements stop
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

# using the Image Generator
datagen.fit(Train_X)
history = model.fit(datagen.flow(Train_X,Train_Y, batch_size=64),
            steps_per_epoch=len(Train_Y) / 64, epochs=200, validation_data=(Test_X, Test_Y), callbacks = [es]
            )

import matplotlib.pyplot as plt

plt.title('Accuracy')
plt.plot(history.history['accuracy'], color='red', label='train')
plt.plot(history.history['val_accuracy'], color='blue', label='val')

plt.title('Loss Function')
plt.plot(history.history['loss'], color='red', label='train')
plt.plot(history.history['val_loss'], color='blue', label='test')



_, accuracy = model.evaluate(Test_X, Test_Y)
print('Accuracy:', accuracy)