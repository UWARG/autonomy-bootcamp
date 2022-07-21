"""
This is a starter file to get you going. You may also include other files if you feel it's necessary.

Make sure to follow the code convention described here:
https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions

Hints:
* The internet is your friend! Don't be afraid to search for tutorials/intros/etc.
* We suggest using a convolutional neural network.
* TensorFlow Keras has the CIFAR-10 dataset as a module, so you don't need to manually download and unpack it.
"""

# Import whatever libraries/modules you need

from ast import arg
import cv2
import matplotlib
from matplotlib import style
import matplotlib.pyplot
import numpy as np
import os
import tensorflow as tf
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from re import L


# Your working code here
REBUILD_DATA = False

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        x = torch.randn(32, 32).view(-1, 1, 32, 32)
        self._to_linear = None
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 10)
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x), (2, 2)))
        x = F.max_pool2d(F.relu(self.conv2(x), (2, 2)))
        x = F.max_pool2d(F.relu(self.conv3(x), (2, 2)))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    def forward(self, x):
        

