"""
Provides helper functions to execute a training run
"""

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import Model


def train(model: Model, train_ds: Dataset, val_ds: Dataset, epochs: int) \
        -> tf.keras.callbacks.History:
    """
    Trains `model` on `train_ds` for `epochs`, evaluating each epoch on `val_ds`
    Returns the History of the training run
    """
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs)
