from keras import Sequential
from keras.datasets import cifar10
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import to_categorical


def load_dataset():
    (train_X, train_y), (test_X, test_y) = cifar10.load_data()
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    return train_X, train_y, test_X, test_y


def prepare_pixels(train, test):
    train_norm = train.astype("float32") / 255.0
    test_norm = test.astype("float32") / 255.0

    return train_norm, test_norm


def def_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)),
        # VGG 1
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),  # VGG 2
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),  # VGG 3
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Flatten(),  # Dense
        Dense(128, activation='relu', kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    opt = SGD(lr=0.001, momentum=0.9)  # compile
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def fit_eval(model, train, test):
    train_X, train_y = train
    test_X, test_y = test
    history = model.fit(train_X, train_y, epochs=100, batch_size=64, validation_data=(test_X, test_y), verbose=1)
    _, acc = model.evaluate(test_X, test_y, verbose=1)
    return history, acc


def run_test():
    train_X, train_y, test_X, test_y = load_dataset()
    train_X, test_X = prepare_pixels(train_X, test_X)

    train, test = (train_X, train_y), (test_X, test_y)
    model = def_model()

    history, acc = fit_eval(model, train, test)
    history.to_csv("history.csv")

    model.save('model.h5')

if __name__ == "__main__":
    run_test()