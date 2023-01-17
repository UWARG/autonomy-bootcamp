from tensorflow.keras import Model
from tensorflow.data import Dataset

def train(model: Model, train_ds: Dataset, val_ds: Dataset, epochs: int):
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs)
