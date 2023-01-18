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
import argparse
import os
import pickle
from typing import Dict

import tensorflow as tf
import yaml

from dataset import prep_dataset
from model import MyCNN
from trainer import train


def setup_gpu():
    """
    Allow GPU memory growth
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as err:
            print(err)

def save_results(history: tf.keras.callbacks.History, cfg: Dict):
    """
    Saves a pickle of `history` to `cfg['out_dir']/cfg['exp_name']/history.pickle`
        and a yaml of `cfg` to `cfg['out_dir']/cfg['exp_name']/cfg.yml`
    """
    out_dir = cfg['out_dir']
    exp_name = cfg['exp_name']
    history_file = f"{out_dir}/{exp_name}/history.pickle"
    cfg_file = f"{out_dir}/{exp_name}/cfg.yml"
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, 'wb') as history_obj:
        pickle.dump(history.history, history_obj)
    with open(cfg_file, 'w', encoding='utf8') as cfg_obj:
        yaml.dump(cfg, cfg_obj)


def main(cfg: Dict):
    """
    Does a training run based on `cfg`, and saves the results to files
    """
    setup_gpu()
    with tf.device('/GPU:0'):
        train_ds, val_ds, ds_info = prep_dataset(cfg['data'])
        num_classes = ds_info.features['label'].num_classes
        model = MyCNN(out_size=num_classes, **cfg['model'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        **cfg['train'])
    history = train(model, train_ds, val_ds, cfg['epochs'])
    save_results(history, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='path to the configuration file')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf8') as config_file:
        load_cfg = yaml.safe_load(config_file)
    main(load_cfg)
