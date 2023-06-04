"""
This is a starter file to get you going. You may also include other files if you feel it's necessary.

Make sure to follow the code convention described here:
https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions

Hints:
* The internet is your friend! Don't be afraid to search for tutorials/intros/etc.
* We suggest using a convolutional neural network.
* TensorFlow Keras has the CIFAR-10 dataset as a module, so you don't need to manually download and unpack it.
"""

# Import Libraries and Modules
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms # This is a module that provides a set of functions that are used for image transformations and useful for data preprocessing
import matplotlib.pyplot as plt
import torch.nn as nn # Provides a set of classes and modules for building/training neural networks
import torch.nn.functional as F # Provides a set of functions that are commonly used in neural network operations such as Sigmoid and Hyperbolic Tangent activation functions
import torch.optim as optim