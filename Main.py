"""
This is a starter file to get you going.
You may also include other files if you feel it's necessary.

Make sure to follow the code convention described here:
https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions

Hints:
* The internet is your friend! Don't be afraid to search for tutorials/intros/etc.
* We suggest using a convolutional neural network.
* TensorFlow Keras has the CIFAR-10 dataset as a module,
    so you don't need to manually download and unpack it.
"""

import tensorflow as tf
import matplotlib.pyplot as plt

def clean_input(*series):
    """Preprocess data to be fed to neural network

    Returns:
        Collection: collection of data to process
    """
    return (s/255.0 for s in series)

def build_model():
    """Build neural network

    Returns:
        tf.keras.Model: Neural Network
    """
    return tf.keras.models.Sequential([tf.keras.layers.Conv2D(3, 5),
                                       tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
                                       tf.keras.layers.Dense(64, activation='relu'),
                                       tf.keras.layers.Dense(32, activation='relu'),
                                       tf.keras.layers.Dropout(0.2),
                                       tf.keras.layers.Dense(10)])

def compile_model(model):
    """Compile model

    Args:
        model (tf.keras.Model): Uncompiled model
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

if __name__ == "__main__":

    EPOCH_COUNT = 10

    # prepare data
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = clean_input(x_train, x_test)

    # prepare model
    image_classifier = build_model()
    compile_model(image_classifier)

    # train model
    train_history = image_classifier.fit(
        x_train, y_train, epochs=EPOCH_COUNT, validation_data=(x_test, y_test)
    )

    # visualize training loss
    epoch_range = range(EPOCH_COUNT)
    plt.plot(epoch_range, train_history.history['loss'], label="training loss")
    plt.plot(epoch_range, train_history.history['val_loss'], label="validation loss")
    plt.legend(loc='lower right')
    plt.title("Loss")
    plt.savefig("loss.jpg")
