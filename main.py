"""
## CIFAR 10 Dataset Classification
Author: Nihal Potdar
"""

# Import the required libraries and the dataset
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

"""Let's import the dataset and analyze its properties using the using keras.datasets api"""

(xTrain, yTrain), (xTest, yTest) = keras.datasets.cifar10.load_data()
def num_examples(xTrain, set_name: str):
  print(f"There are {len(xTrain)} examples in the {set_name} set with shape {np.shape(xTrain[0])}")
num_examples(xTrain, "training set")
num_examples(xTest, "testing set")
# Let's convert the targets to match the format of the output of the network (ie: class i is 1 if it is the target for that input)

"""We see from above that there are 3 channels in the image (rgb) with dimensions of 32*32 <br/>
We also note from observation of the data, that the data contains 10 classes of real-world objects</br>

For the data processing, let's normalize the values of the pixels to make the computations more efficient <br/>
We cast the numpy array as float and then divide by 255 (pixel values range fom 0-255)
"""

def normalize_pixels(xTrain, xTest):
  xTrain = xTrain.astype(float)
  xTest = xTest.astype(float)
  xTrain = xTrain/255.0
  xTest = xTest/255.0
  return xTrain, xTest
xTrain, xTest = normalize_pixels(xTrain, xTest)
# print(xTrain[0]) # print out the pixels to validate their values

"""The pixel values are now normalizes between 0 and 1
Let's introduce some image augmentation to increase the number of examples that we have available. Introducing Image Augmentation will also make the model more robust with more extensive cases that it normally wouldn't encounter

Below, we construct the model <br/>
We are using a CNN-based model where the CNN (convolution) iterates over the inputs with a kernel (of size (3,3)) and stride (default stride of (1,1) below), multiplying the inputs by a sequence of weights which will be updated by back-propagation. For each channel, the convolution will be repeated (for a 2D convolution, the number of channels will not change).

After the convolution layer, we apply a max pooling layer which iterates over the inputs with a filter and replaces the value of the centered square with the max of all the squares that are included. We add more convolution layers to further reduce the size of the feature space. The CNN layers are organized with increasing filter size to organize it in such a way that the earlier layers extract the higher level information and any lower levels more complex information.

We then flatten the inputs to be passed into the dense layers, to combine all the channels into one.

The dense layers then learn a mapping from the flattened feature space to the space of the outputs, producing a probability distribution over the out classes with a softmax. Applying the 'ReLU' activation function on each output layer will speed up training and prevent the vanishing gradient problem. Data Augmentation and dropout are used as techniques to prevent overfitting. Data Augmentation introduces more noise to the examples to make it harder for the model, and dropout randomly turns off some neurons to force other neurons to do more work.
"""

data_aug = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    # keras.layers.experimental.preprocessing.RandomRotation(0.2)
])

NUM_CLASSES = 10
model = keras.Sequential([
    data_aug,
    keras.layers.Conv2D(32, 3, padding="same", activation='relu', input_shape=[32,32,3]),
    keras.layers.MaxPool2D(2, 2, padding="valid"),
    keras.layers.Conv2D(64, 3, padding="same", activation='relu'),
    keras.layers.MaxPool2D(2, 2, padding="valid"),
    keras.layers.Conv2D(128, 3, padding="same", activation='relu'),
    keras.layers.MaxPool2D(2, 2, padding="valid"),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(NUM_CLASSES, activation="softmax")
])
# model.summary()

"""Let's compile the model with the SparseCategoricalCrossentropy loss and adam optimizer"""

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

"""Let's train the model for 30 epochs with shuffling among the input batches every epoch"""

history = model.fit(
    x=xTrain,
    y=yTrain,
    validation_data=(xTest, yTest),
    shuffle=True,
    batch_size=32,
    epochs=30
)

"""## Results
Comparisons (after 5 epochs)
Baseline (1 CNN layer) - val accuracy=56.32% <br/>
2 CNN Layer model (32, 16) - val accuracy=61.04% <br/>
2 CNN Layer model (128, 63) - val accuracy=62.03% <br/>
3 CNN Layer model (32, 64, 128) - val accuracy=67.42% <br/>

We observe that model 3 works best, let's train for longer and introduce over-fitting techniques
(15 epochs) <br/>
(no overfitting techniques) 3 CNN Layer model (32, 64, 128) - val accuracy=72.18% <br/>
(dropout+data aug) 3 CNN Layer model (32, 64, 128) - val accuracy= 57.61%<br/>
(droout only) - val ccuracy = 75.73%
(dropout+linear layer dimension change) - val accuracy=72.59% <br/>
(dropout + data aug, 30 epochs) - val accuracy = 75.84%
"""

acc = history.history["accuracy"]
valAcc = history.history["val_accuracy"]

loss = history.history["loss"]
valLoss = history.history["val_loss"]

plt.title("training accuracy vs validation accuracy")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(acc)
plt.plot(valAcc)
plt.show()

plt.title("training loss vs validation loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(loss)
plt.plot(valLoss)
plt.show()
