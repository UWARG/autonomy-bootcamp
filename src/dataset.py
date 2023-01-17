import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import DatasetInfo
from tensorflow.data import Dataset
from tensorflow.keras import layers
from typing import Tuple, Dict
from functools import partial

def __construct_image_pipeline(pre: Dict, mean: float, std: float, training: bool):
    seq = []
    seq.append(layers.Rescaling(1.0, offset=-mean))
    seq.append(layers.Rescaling(1. / std))
    if training and 'random_flip' in pre:
        seq.append(layers.RandomFlip(pre['random_flip']))
    if training and 'random_rotation' in pre:
        seq.append(layers.RandomRotation(pre['random_rotation']))
    return tf.keras.Sequential(seq)

def __mean_std(ds: Dataset) -> Dict[str, float]:
    # Define a function to compute mean and std of each image
    def compute_stats(image, _):
        image = tf.cast(image, tf.float32)
        mean, variance = tf.nn.moments(image, axes=[0, 1, 2])
        return mean, variance

    # Compute mean and std of the entire dataset
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    # In the context of the tf.data.Dataset.reduce function, 
    #   the x and y lambda variables are used to represent 
    #   the current element and the accumulator respectively.

    def reduce_func(x, y):
        return (x[0] + y[0], x[1] + y[1])

    mean, variance = ds.map(compute_stats).reduce((0., 0.), reduce_func)
    mean = mean / len(ds)
    std = tf.sqrt(variance / len(ds))
    return {'mean': mean, 'std': std}

def __preprocess_sample(image, label, image_pipeline: tf.keras.Sequential):
    return image_pipeline(image), label

def prep_dataset(data_cfg: Dict) -> Tuple[Dataset, Dataset, DatasetInfo]:
    (train_ds, val_ds), meta = tfds.load("cifar10", split=[data_cfg['train_split'], data_cfg['val_split']], with_info=True, as_supervised=True)
    mean_std_vals = __mean_std(train_ds)
    train_image_pipeline = __construct_image_pipeline(data_cfg, training=True, **mean_std_vals)
    val_image_pipeline = __construct_image_pipeline(data_cfg, training=False, **mean_std_vals)
    train_ds = train_ds.map(partial(__preprocess_sample, image_pipeline=train_image_pipeline))
    val_ds = val_ds.map(partial(__preprocess_sample, image_pipeline=val_image_pipeline))
    train_ds = train_ds.shuffle(data_cfg['shuffle']).batch(data_cfg['batch_size'])
    val_ds = val_ds.batch(data_cfg['batch_size'])
    return train_ds, val_ds, meta