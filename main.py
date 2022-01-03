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

# imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
from google.colab import files
from keras.preprocessing import image

# Your working code here
# downloading the training, testing images and labels using the tf load_data() method
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# now, we need to normalize the images to be able to pass them through the CNN (all values should be 0 < x < 1)
# normalizing training images
x_train = x_train / 255
# normalizing testing images
x_test = x_test / 255


model = tf.keras.models.Sequential([
    # input shape is 32x32x3 since the size of each image is 32x32 pixels, with 3 colours
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # dropout layer(s) to reduce overfitting
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
    # the last layer has 10 neurons since we have a total of 10 classes we need to classify
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

# compiling the model with the 'adam' optimizer function, 
# 'sparse_categorical_crossentropy' loss function since 
# we need to classify multiple classes
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(
    x_train,
    y_train,
    epochs = 15,
    verbose=1,
    validation_data=(x_test, y_test),
)


# Retrieve a list of list results on training and test data
# sets for each training epoch

accuracy=history.history['accuracy']
val_accuracy=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(accuracy)) # Get number of epochs

# training and validation accuracy per epoch
plt.plot(epochs, accuracy, 'r', "Training Accuracy")
plt.plot(epochs, val_accuracy, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()


# training and validation loss per epoch
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title('Training and validation loss')



# code block to manually input image into the model and see result

uploaded = files.upload()

for fn in uploaded.keys():
#   save files in a content dir
  path = '/content/' + fn
  img = image.load_img(path, target_size=(32, 32, 3))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])
  if classes[0][0]>0:
    print(fn + " is an airplane")
  elif classes[0][1]>0:
    print(fn + " is an automobile")
  elif classes[0][2]>0:
    print(fn + " is a bird")
  elif classes[0][3]>0:
    print(fn + " is a cat")
  elif classes[0][4]>0:
    print(fn + " is a deer")
  elif classes[0][5]>0:
    print(fn + " is a dog")
  elif classes[0][6]>0:
    print(fn + " is a frog")
  elif classes[0][7]>0:
    print(fn + " is a horse")
  elif classes[0][8]>0:
    print(fn + " is a ship")
  elif classes[0][9]>0:
    print(fn + " is a truck")