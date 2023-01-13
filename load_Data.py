import torch
import torchvision
import torchvision.transforms as transforms
import numpy as pd
import matplotlib.pyplot as plt


tensor_transform = transforms.Compose([transforms.ToTensor()])
# this represents the data that the module with be trained on
training_data = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = tensor_transform)

# This represents the data set that is going to be used to evaluate the trainned module
evaluation_data = torchvision.datasets.CIFAR10(root = './data', train= False, download = True, transform = tensor_transform)

# After reasearch I realized this data set has 10 different objects so I made a dictionary repressenting each object 
different_objects_in_data = {
    0 : 'plane',
    1 : 'car',
    2 : 'bird', 
    3 : 'cat',
    4 : 'deer',
    5 : 'dog',
    6 : 'frog',
    7 : 'horse',
    8 : 'ship',
    9 : 'truck',
}

# this takes in each dataset (train and evaluation and sets a batch size)
# shuffle is set to true for train as if the module is trained on the same module it woould mostly like be in accurate 
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size = 4, shuffle = True)
Evaluation_dataloader = torch.utils.data.DataLoader(evaluation_data, batch_size = 4, shuffle = False)

# This is an iterable to select different batches of the same size
train_iter = iter(training_dataloader)

print("ff")