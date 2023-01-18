"""
Contains helper functions to prepare the CIFAR-10 dataset
"""

from functools import partial
from typing import Dict, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.data import Dataset
from tensorflow.keras import layers
from tensorflow_datasets.core import DatasetInfo

def __construct_image_pipeline(pre: Dict, mean: float, std: float, training: bool) \
    -> tf.keras.Sequential:
    """
    Constructs an image processing pipeline according to the key-value pairs in `pre`
    """
    seq = []
    seq.append(layers.Rescaling(1.0, offset=-mean))
    seq.append(layers.Rescaling(1. / std))
    if training and 'random_flip' in pre:
        seq.append(layers.RandomFlip(pre['random_flip']))
    if training and 'random_rotation' in pre:
        seq.append(layers.RandomRotation(pre['random_rotation']))
    if training and 'random_zoom' in pre:
        seq.append(layers.RandomZoom(pre['random_zoom']))
    return tf.keras.Sequential(seq)

def __mean_std(dataset: Dataset) -> Dict[str, float]:
    """
    Computes the mean and std of all images in `dataset`
    Returns them as a named Dict
    """
    # Define a function to compute mean and std of each image
    def compute_stats(image, _):
        image = tf.cast(image, tf.float32)
        mean, variance = tf.nn.moments(image, axes=[0, 1, 2])
        return mean, variance

    # Compute mean and std of the entire dataset
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    mean, variance = dataset.map(compute_stats).reduce((0., 0.),
        lambda x, y: (x[0] + y[0], x[1] + y[1]))
    mean = mean / len(dataset)
    std = tf.sqrt(variance / len(dataset))
    return {'mean': mean, 'std': std}

def __preprocess_sample(image: tf.Tensor, label: int, image_pipeline: tf.keras.Sequential) \
        -> Tuple[tf.Tensor, int]:
    return image_pipeline(image), label

def prep_dataset(data_cfg: Dict) -> Tuple[Dataset, Dataset, DatasetInfo]:
    """
    Creates a training and validation dataset according to the specification in `data_cfg`
    Supports:
        - Splitting
        - Image augmentation
        - Sample repeating
        - Shuffling
        - Batching
    """
    # Load base dataset splits
    (train_ds, val_ds), meta = tfds.load("cifar10",
        split=[data_cfg['train_split'],
        data_cfg['val_split']],
        with_info=True, as_supervised=True)

    # Perform image augmentaton
    mean_std_vals = __mean_std(train_ds)
    train_image_pipeline = __construct_image_pipeline(data_cfg, training=True, **mean_std_vals)
    val_image_pipeline = __construct_image_pipeline(data_cfg, training=False, **mean_std_vals)
    train_ds = train_ds.map(partial(__preprocess_sample, image_pipeline=train_image_pipeline))
    val_ds = val_ds.map(partial(__preprocess_sample, image_pipeline=val_image_pipeline))

    # Repeat, shuffle, batch
    train_ds = train_ds \
        .shuffle(data_cfg['shuffle']) \
        .batch(data_cfg['batch_size']) \
        .repeat(data_cfg['repeat'])
    val_ds = val_ds.batch(data_cfg['batch_size'])

    return train_ds, val_ds, meta
