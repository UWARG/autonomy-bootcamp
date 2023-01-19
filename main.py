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

import ssl

import torch
from torchvision import datasets, transforms

from model import Model

# Your working code here

def main():
    device = get_device()
    train_set, test_set = get_datasets()
    train_loader, test_loader = data_loaders(train_set, test_set)
    m = Model(device, train_set, test_set, train_loader, test_loader)
    m.build_cnn()
    m.set_loss_opt()
    m.train_model()
    m.plot_results()

# train on GPU if available
def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# get the training and test sets
def get_datasets():
    # standard cast into Tensor and pixel values normalization in [-1, 1] range
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # extra transfrom for the training data, in order to achieve better performance
    # randomly flip the image horizontally with a probability of 0.5
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
        transforms.RandomHorizontalFlip(), 
    ])
    
    #permissions for downloading CIFAR10 dataset manually
    ssl._create_default_https_context = ssl._create_unverified_context

    train_set = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    return train_set, test_set

# get the training and test loaders
def data_loaders(train_set, test_set):
    # number of subprocesses to use for data loading
    num_workers = 0
    
    size = 32
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=size, 
        shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader

if __name__ == "__main__":
    main()