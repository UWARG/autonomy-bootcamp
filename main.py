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

import numpy as np

# Your working code here

from json import load
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# the classes of data which are possible in CFAIR-10 dataset
classes = ['airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(classes)


"""
Function: load_data()
dowloads, transforms and creates a dataloader for the data

Paramteres: 
None

Return: 
train_loader, test_loader (which can be enumerated)
"""


def load_data():
    BATCH_SIZE = 64
    torch.manual_seed(0)

    # Here we convert RGB values to tensors of size 3*32*32
    # We also normalize the tensors for faster convergance during optimization
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                                                         0.2470, 0.2435, 0.2616])
                                    ])

# these are the transformed images
    train_set = torchvision.datasets.CIFAR10(
        root='./root', download=True, train=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(
        root='./root', download=True, train=False, transform=transform)

    # We shuffle the loader for better accuracy
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader


"""
Function: Model()
creates a CNN model for image classification

Paramteres: 
None

Return: 
model
(It is an object of extended classe nn.Module)
"""


def Model():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            # this is poling layer that is applied after each convolution layer
            self.pool = nn.MaxPool2d(2, 2)
            # Convolution Layer #1 for feature extraction
            self.covn1 = nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, padding=1)
            # Convolution Layer #2 for feature extraction
            self.conv2 = nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, padding=1)
            # Convolution Layer #3 for feature extraction
            self.conv3 = nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1)
            # Convolution Layer #4 for feature extraction
            self.conv4 = nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1)
            # Convolution Layer #5 for feature extraction
            self.conv5 = nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1)
            # Batch normalization for better accuracy
            self.batch1 = nn.BatchNorm2d(num_features=16)
            self.batch2 = nn.BatchNorm2d(num_features=32)
            self.batch3 = nn.BatchNorm2d(num_features=64)
            self.batch4 = nn.BatchNorm2d(num_features=128)
            self.fc1 = nn.Linear(256, 120)  # Dense layer #1 for Non-linearity
            self.fc2 = nn.Linear(120, 84)  # Dense layer #2 for Non-linearity
            self.fc3 = nn.Linear(84, 10)  # Dense layer #3 for Non-linearity

        def forward(self, x):
            x = self.pool(F.relu(self.covn1(x)))
            x = self.batch1(x)
            x = self.pool(F.relu(self.conv2(x)))
            x = self.batch2(x)
            x = self.pool(F.relu(self.conv3(x)))
            x = self.batch3(x)
            x = self.pool(F.relu(self.conv4(x)))
            x = self.batch4(x)
            x = self.pool(F.relu(self.conv5(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

            return x

    net = Net()

    return net


"""
Function: loss_opt()
defines a loss function and optimizer for image classification

Paramteres: 
model

Return: 
loss_function, optimizer
"""


def loss_opt(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return criterion, optimizer


train_losses = []
train_accu = []

"""
Function: train()
For a single epoch evaluates the model on training dataset, calculates the loss, and updates weights and biases

Paramteres: 
epoch, model, trainloader, lossFunction, optimizer 

Return: 
None
"""


def train(epoch, model, trainloader, lossFunction, optimizer):
    print('\nEpoch : %d' % epoch)

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    # Determining training loss from model predictions
    for data in tqdm(trainloader):

        inputs, labels = data
        outputs = model(inputs)
        loss = lossFunction(outputs, labels)

        optimizer.zero_grad()
        loss.backward()                                 # computes the gradient
        # use the gradient to tweak the parameters
        optimizer.step()

        # converts the loss into  a float datatype and adds it to running_loss
        running_loss += loss.item()

        # the class neuron with the highest activation is chosen as the prediction
        _, predicted = outputs.max(1)
        # adds to the total samples
        total += labels.size(0)
        # compares the labels to the predicted, adds the amount to correct and transalates tensor to a number
        correct += predicted.eq(labels).sum().item()

    # Defining training loss
    train_loss = running_loss/len(trainloader)
    accu = 100.*correct/total

    # Printing model
    train_accu.append(accu)
    train_losses.append(train_loss)
    print('Train Loss: %.3f | Accuracy: %.3f' % (train_loss, accu))


eval_losses = []
eval_accu = []


"""
Function: test()
For a single epoch evaluates the model on testing dataset, and calculates the running loss. 

Paramteres: 
epoch, model, testloader, lossFunction 

Return: 
None
"""


def test(epoch, model, testloader, lossfunction):
    print('\n Test Epoch : %d' % epoch)
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data

            outputs = model(images)
            _, predicted = outputs.max(1)

            loss = lossfunction(outputs, labels)
            running_loss += loss.item()

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss/len(testloader)
    accu = 100*(correct/total)

    eval_accu.append(accu)
    eval_losses.append(test_loss)

    print('Test Loss: %.3f | Accuracy: %.3f' % (test_loss, accu))


"""
Function: train_model()
Trains the model for EPOCHS number of epochs, and calculates loss and accuracy of the model on training and testing set.

Paramteres: 
model, trainloader, testloader, lossfunction, optimizer, EPOCHS

Return: 
None
"""


def train_model(model, train_loader, test_loader, lossfunction, optimizer, EPOCHS):

    for epoch in range(EPOCHS):
        train(epoch, model, train_loader, lossfunction, optimizer)
        test(epoch, model, test_loader, lossfunction)


"""
Function: plot_loss()
plots the training and testing loss. 

Paramteres: 
None

Return: 
None
"""


def plot_loss():
    plt.plot(train_losses, '-o')
    plt.plot(eval_losses, '-o')
    plt.title('Train vs Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Test'])
    plt.savefig("loss.png")
    plt.close()


"""
Function: plot_accuracy()
plots the training and testing accuracy. 

Paramteres: 
None

Return: 
None
"""


def plot_accuracy():
    plt.plot(train_accu, '-o')
    plt.plot(eval_accu, '-o')
    plt.title('Train vs Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Valid'])
    plt.savefig("accuracy.png")
    plt.close()


"""
Function: main()
Runs every function defined above

Paramteres: 
None

Return: 
None
"""


def main():
    train_loader, test_loader = load_data()
    model = Model()
    loss_function, optimizer = loss_opt(model)
    EPOCH = 7
    train_model(model, train_loader, test_loader,
                loss_function, optimizer, EPOCH)

    plot_loss()
    plot_accuracy()


if __name__ == "__main__":
    main()