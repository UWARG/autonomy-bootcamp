import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

def unpickle(file):
    import pickle
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    return dict

class_names = ["airplane", "automobile", "bird","cat", "deer", "dog", "frog", "horse", "ship", "truck"]										

training_batch = {"l": unpickle("CIFAR10/data_batch_1")[b'labels'], "d": unpickle("CIFAR10/data_batch_1")[b'data']}

for i in range(2,6):
    training_batch["l"] += unpickle(f"CIFAR10/data_batch_{i}")[b'labels']

    training_batch["d"] = np.concatenate((training_batch["d"],unpickle(f"CIFAR10/data_batch_{i}")[b'data']))


test_batch = {"l": unpickle("CIFAR10/test_batch")[b'labels'], "d": unpickle("CIFAR10/test_batch")[b'data']}

# Diving all values by 255.0 to normalize data and resizing data from a 1D to a 32x32 Matrix

training_batch["d"] = np.true_divide(training_batch["d"], 255.0).reshape(len(training_batch["d"]), 32, 32, 3)
test_batch["d"] = np.true_divide(test_batch["d"], 255.0).reshape(len(test_batch["d"]), 32, 32, 3)

model = models.Sequential()

model.add(layers.Conv2D(64, (3,3), activation = "relu", input_shape = (32,32,3), padding = "same"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation = "relu", padding = "same"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation = "relu", padding = "same"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation = "relu", padding = "same"))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())
model.add(layers.Dropout(0.3))

model.add(layers.Dense(1024, activation = "relu"))

# # THE FINAL LAYER:
# #   10, corresponding to the 10 classes of the CIFAR-10 Dataset
# model.add(layers.Dense(10, activation = "softmax"))

model.compile(optimizer="adam", loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(training_batch["d"], training_batch["l"], epochs = 6)

test_loss, test_acc = model.evaluate(test_batch["d"], test_batch["l"])

print(test_acc)