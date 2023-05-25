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

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms

# Your working code here
"""
CIFAR-10 has 10 classes of images, and each image is 32 x 32 pixels and has 3 color values. 
Training data set size is 50,000 images, and test dataset size is 10,000 images.
"""
# Transforming the images
"""
- transforms.Compose() takes in a sequence of transformations that are to be applied to the input data.
- transforms.ToTensor() converts the input images to tensors with the shape of channels, height, and width. 
- transforms.Normalize() takes in two arguments, mean and std. Since CIFAR10 is a 3 channel dataset, (0.5, 0.5, 0.5)
represents the mean values and standard deviation for the RGB channels. 
"""

transform_data = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Loading the testing and training data sets
# Batch size is the number of samples processed in one iteration, ie, forward and backward pass.
batch_size = 16

# Loading the dataset
train_set = torchvision.datasets.CIFAR10(root='/data',
                                         train=True,
                                         download=True,
                                         transform=transform_data)

train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=batch_size,
                                           shuffle=True)

test_set = torchvision.datasets.CIFAR10(root='/data',
                                        train=False,
                                        download=True,
                                        transform=transform_data)

test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=batch_size,
                                          shuffle=True)


# Developing a Neural Network Architecture as a class
# Subclassing nn.Module
class Network (nn.Module):
    def __init__(self):
        # Network inherits its functions and behavior from super class nn.Module
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 6 * 6, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Instance of the neural network class "Network"
net = Network()

train_loss = []
train_accu = []

test_loss = []
test_accu = []

# Defining the loss/cost function
"""
CrossEntropyLoss() is a loss function that computes the SoftMax Activation and then computes the cross entropy loss.
The SoftMax Activation takes the output of the model in the form of raw data for each class and converts it 
to a probability distribution over the classes. It normalises the probabilities to add up to 1.
The cross entropy loss function measures the dissimilarity between the predicted probability and the true labels. 

"optimizer" is an optimizer object used to train the neural network using SGD--stochastic gradient descent.
"lr" is the learning rate and it determines the step size at each iteration of the optimizatio process
"momentum" helps speed up the optimization process
"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# Defining a function to training the neural network
"""
The models parameters are updated based on the feedback from the loss functions.
The model performs forward propagation to make predictions, calculates the loss between 
the predicted and actual values, and then performs backward propagation to update the models
weights and biases using an optimization algo (optimizer variable from line 106).
"""


def train(network, data_loader, criterion, optimizer):
    dataset_size = len(data_loader.dataset)
    num_batches = len(data_loader)
    net.train()
    loss_in_one_epoch = 0
    correct_pred = 0

    for i, data in enumerate(data_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        model_outputs = net(inputs)
        loss = criterion(model_outputs, labels)
        loss.backward()
        optimizer.step()

        loss_in_one_epoch += loss.item()
        correct_pred += (model_outputs.argmax(1) ==
                         labels).type(torch.float).sum().item()

        if i % 2000 == 1999:
            print('[%d, %5d] Training loss: %.3f' %
                  (epoch + 1, i + 1, loss_in_one_epoch / 2000))

    loss_in_one_epoch /= num_batches
    correct_pred /= dataset_size
    accuracy = 100 * correct_pred

    print(
        f"Train Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {loss_in_one_epoch:>8f} \n")
    train_accu.append(correct_pred)
    train_loss.append(loss_in_one_epoch)


# Defining a function to test the neural network
def test(network, data_loader, criterion):
    dataset_size = len(data_loader.dataset)
    num_batches = len(data_loader)
    net.eval()
    loss = 0
    correct_pred = 0

    with torch.no_grad():
        for data in data_loader:
            images, labels = data

            model_output = network(images)

            batch_loss = criterion(model_output, labels).item()
            loss += batch_loss
            correct_pred += (model_output.argmax(1) ==
                             labels).type(torch.float).sum().item()

        loss /= num_batches
        correct_pred /= dataset_size
        accuracy = 100 * correct_pred
        print(
            f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {loss:>8f} \n")
        test_accu.append(correct_pred)
        test_loss.append(loss)

# Definig functions to graph the train vs test loss and accuracy


def graph_accu():
    plt.plot(train_accu, '-o')
    plt.plot(test_accu, '-o')
    plt.title('Train vs Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Valid'])


def graph_loss():
    plt.plot(train_loss, '-o')
    plt.plot(test_loss, '-o')
    plt.title('Train vs Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Test'])


# Epochs is the number of training iterations the neural network undertakes
epoch = 10

for epochs in range(epoch):
    print(f"Epoch {epochs+1}\n-------------------------------")
    train(net, train_loader, criterion, optimizer)
    test(net, test_loader, criterion)

print("Training and Testing completed")
graph_loss()
graph_accu()
