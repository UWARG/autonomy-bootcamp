import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

def plot_sample(X, y, index):
    plt.figure(figsize = (15, 2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])
    plt.show()

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

y_train = y_train.reshape(-1,)

X_train = X_train/255
X_test = X_test/255

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
model_history = cnn.fit(X_train, y_train, epochs=10)
cnn.evaluate(X_test, y_test)
y_pred = cnn.predict(X_test)
y_class = [np.argmax(element) for element in y_pred]

def plot_loss():
    plt.plot(model_history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Test'])
    plt.savefig('loss_mazin.png')
    plt.close()

def plot_accuracy():
    plt.plot(model_history.history['accuracy'], label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Test'])
    plt.savefig('accuracy_mazin.png')
    plt.close()

plot_accuracy()
plot_loss()