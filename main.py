"""
This is a starter file to get you going. You may also include other files if you feel it's necessary.

Make sure to follow the code convention described here:
https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions

Hints:
* The internet is your friend! Don't be afraid to search for tutorials/intros/etc.
* We suggest using a convolutional neural network.
* TensorFlow Keras has the CIFAR-10 dataset as a module, so you don't need to manually download and unpack it.
"""

# Importing the required libraries/modules
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

# Data classes for training and testing
CLASSES = ['airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
EVAL_LOSSES = []
EVAL_ACCU = []
TRAIN_LOSSES = []
TRAIN_ACCU = []


def device_classification():
    return (torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))


def load_data():
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=TRANSFORM)

    trainLoader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                              shuffle=True, num_workers=0)

    test_set = torchvision.datasets.CIFAR10('data', train=False,
                                            download=True, transform=TRANSFORM)

    testLoader = torch.utils.data.DataLoader(test_set, batch_size=32,
                                             shuffle=False, num_workers=0)
    return trainLoader, testLoader


def cnn(device):
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(64),

        # Increasing the depth to 64
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(128),

        # Increasing the depth to 128
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        # Flattening the model
        nn.Flatten(),
        nn.Linear(256*4*4, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )

    model.to(device)
    return model


def loss_opt(model):
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return lossFunction, optimizer


def train(model, trainloader, device, lossFunction, optimizer):
    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for data in tqdm(trainloader):

        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = lossFunction(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss/len(trainloader)
    accu = 100.*correct/total

    TRAIN_ACCU.append(accu)
    TRAIN_LOSSES.append(train_loss)
    print('Train Loss: %.3f | Accuracy: %.3f' % (train_loss, accu))


def test(model, testLoader, device, lossFunction):
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(testLoader):
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)

            loss = lossFunction(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss/len(testLoader)
    accu = 100.*correct/total

    EVAL_LOSSES.append(test_loss)
    EVAL_ACCU.append(accu)

    # Printing model testing progress
    print('Test Loss: %.3f | Accuracy: %.3f' % (test_loss, accu))

# Runs train_model
def train_model(model, trainLoader, testLoader, device, lossFunction, optimizer):
    for epoch in range(1, 10):
        print('\nEpoch : %d'%epoch)
        train(model, trainLoader, device, lossFunction, optimizer)
        test(model, testLoader, device, lossFunction)

def plot_loss():
    plt.plot(TRAIN_LOSSES, '-o')
    plt.plot(EVAL_LOSSES, '-o')
    plt.title('Train vs Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Test'])
    plt.savefig("loss.png")
    plt.close()

# Defining plot
def plot_accuracy():
    plt.plot(TRAIN_ACCU, '-o')
    plt.plot(EVAL_ACCU, '-o')
    plt.title('Train vs Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Valid'])
    plt.savefig("accuracy.png")
    plt.close()

def main():

    # Check for training on GPU
    device = device_classification()

    # Load data
    trainLoader, testLoader = load_data()

    # Run model
    model = cnn(device)

    # Optimize
    lossFunction, optimizer = loss_opt(model)

    # Train model
    train_model(model, trainLoader, testLoader, device, lossFunction, optimizer)

    # Loss plot
    plot_loss()

    # Accuracy plot
    plot_accuracy()


# Run all functions
main()

