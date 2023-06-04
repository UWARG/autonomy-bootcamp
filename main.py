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

# Load and Normalize the CIFAR-10 Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Defines a sequence of transformations
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # Creates an object train_set that represents the CIFAR-10 dataset with transform applied to it and then stored in the ./data directory.
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) # Creates an object test_set that represents the CIFAR-10 dataset with transform applied to it and then stored in the ./data directory.

batch_size = 4
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

# Define Classes for Images in the Dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')