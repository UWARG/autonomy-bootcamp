import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split


EPOCHS = 1
IMG_WIDTH = 32
IMG_HEIGHT = 32
NUM_CATEGORIES = 10


def load_data():
    print("Loading data...")
    cifar10 = tf.keras.datasets.cifar10

    (trainImages, trainLabels), (testImages, testLabels) = cifar10.load_data()

    # Normalize pixel values to between 0 and 1
    trainImages, testImages = trainImages / 255.0, testImages / 255.0

    # Convert trainLabels and testLabels from integers to binary class matrix
    trainLabels = tf.keras.utils.to_categorical(trainLabels)
    testLabels = tf.keras.utils.to_categorical(testLabels)

    # Split training data into train and validation sets
    trainImages, valImages, trainLabels, valLabels = train_test_split(trainImages, trainLabels, test_size=0.2)

    return trainImages, trainLabels, valImages, valLabels, testImages, testLabels


def build_model():
    model = tf.keras.models.Sequential([
        # Create 32 feature maps with a 3x3 kernel matrix
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        # Pick the pixel from 2x2 patches which has the maximum value
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Drop some of the nodes to ensure the model isn't overfitting by relying too much on certain nodes
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        # Convert 3D tensor output from pooling layer to 1D vector
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        # Last layer with the same number of nodes as the categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # Configure model for training
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model


def plot_loss(history):
    # Plot loss and validation loss
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0.5, 2])
    plt.legend(loc='lower right')

    plt.show()


if __name__ == '__main__':
    trainImages, trainLabels, valImages, valLabels, testImages, testLabels = load_data()
    model = build_model()
    # Fit the model on the training data, and validate with testing data, by running EPOCHS
    history = model.fit(trainImages, trainLabels, epochs=EPOCHS, validation_data=(valImages, valLabels))
    plot_loss(history)
    model.evaluate(testImages, testLabels)