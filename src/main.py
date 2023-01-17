"""
This is a starter file to get you going. You may also include other files if you feel it's necessary.

Make sure to follow the code convention described here:
https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions

Hints:
* The internet is your friend! Don't be afraid to search for tutorials/intros/etc.
* We suggest using a convolutional neural network.
* TensorFlow Keras has the CIFAR-10 dataset as a module, so you don't need to manually download and unpack it.
"""
from typing import Dict
import argparse
import yaml

import tensorflow as tf
from dataset import prep_dataset
from model import MyCNN
from trainer import train

def setup_GPU():
    # Allow GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def main(cfg: Dict):
    setup_GPU()
    with tf.device('/GPU:0'):
        train_ds, val_ds, ds_info = prep_dataset(cfg['data'])
        num_classes = ds_info.features['label'].num_classes
        model = MyCNN(out_size=num_classes, **cfg['model'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), **cfg['train'])
    history = train(model, train_ds, val_ds, cfg['epochs'])
    print(history)
    print(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='path to the configuration file')
    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        cfg = yaml.safe_load(config_file)
    main(cfg)