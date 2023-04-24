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
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as trfmns
import matplotlib.pyplot as plt


# Your working code here

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform & normalize data 
transform = trfmns.Compose([trfmns.ToTensor(), trfmns.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

# train and testing data
train_losses = []
train_accu = []
test_losses = []
test_accu = []


# Training data (CIFAR) 
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

# Testing data (CIFAR) where train=False
test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# Define batch size of CIFAR data
batch_size = 128

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Create CNN
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fullc1 = nn.Linear(64*4*4, 500)
        self.dropout = nn.Dropout(0.2)
        self.fullc2 = nn.Linear(500, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64*4*4) # -1 automatically computes size of 1st dimension
        x = F.relu(self.fullc1(x))
        x = self.dropout(x)
        x = self.fullc2(x)

        return x

# Create model instance
model = NeuralNetwork().to(device)

# loss and optimizer functions (0.01 is learning rate)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0

    for batch, (X, y) in enumerate(dataloader): # X = images; y = labels
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    train_accu.append(correct)
    train_losses.append(train_loss)
          
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    # disable gradient calculation
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    test_accu.append(correct)
    test_losses.append(test_loss)

# plot loss and accuracy
def plot_losses():
    plt.plot(train_losses, '-o')
    plt.plot(test_losses, '-o')
    plt.title('Train vs Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Test'])
    plt.savefig("loss.png")
    plt.close()

def plot_accu():
    plt.plot(train_accu, '-o')
    plt.plot(test_accu, '-o')
    plt.title('Train vs Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Valid'])
    plt.savefig("accuracy.png")
    plt.close()

# define epochs
epochs = 10

# perform all functions
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")
plot_losses()
plot_accu()





