import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
import cv2
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# four steps of image classification

# 1.loading and data preprocessing

# imports the data from cifar10 database into 2 sets of data, one for training and one for testing
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


#one hot encoding the different images

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

#break the training set into part of it for validation


# each image needs pixel values between 0-1 for better training and currently pixel values are between 0-255 so preprocess the data to be better fit for the model

datagen = ImageDataGenerator(rescale= 1/255, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen.fit(x_train)
# this layer is will be preprocessing the data and will be included in the model for better redability and simplicity

# 2. definining model architecture


#first create a model that will be trained
model = Sequential([

    # neural networks are used as they simulate a human brain with densely packed brain cells
    # convolutional layers are used since these layers are more adept at 2d image classification than standard neural networks
    #convolutional layers take input images and create multiple different outputs called feature maps

    #same padding results in output with the same dimensions as the input
    #activation is the function that is used by the neural network to determine which values are rejected during hidden layer calculation
    layers.Conv2D(32, 3, padding='same', activation='relu'),

    #standardizes data to reduce the number of epochs needed to train neural networks
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    #pooling layer then takes the feature maps and reduces the size and amount of parameters in order to reduce number of calculations
    layers.MaxPooling2D(),
    #dropout removes data to prevent overfitting
    layers.Dropout(0.2),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),
    #flatten layer converts all feature maps into a single dimensional array
    layers.Flatten(),

    #the single dimensional array is then fed into a dense layer which connects every neuron from the previous flattening layer to every neuron in the dense layer
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])


#the optimizer is aiming at making the model more accurate by moving the weighting around
#the loss parameter is the loss function and trys to guide the optimzier around.

model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

# 3. training the model

#an epoch is one lifecycle through the dataset
#steps per epoch is the total amount of data/batch size

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    steps_per_epoch = x_train.shape[0] / 64,
    epochs= 100,
    validation_data=(x_test, y_test)
)


# 4. estimate and visualize training results
#history.history returns the finalized data
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(100)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


model.predict(x_test)