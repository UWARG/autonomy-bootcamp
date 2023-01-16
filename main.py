"""
This is a starter file to get you going. You may also include other files if you feel it's necessary.

Make sure to follow the code convention described here:
https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions

Hints:
* The internet is your friend! Don't be afraid to search for tutorials/intros/etc.
* We suggest using a convolutional neural network.
* TensorFlow Keras has the CIFAR-10 dataset as a module, so you don't need to manually download and unpack it.
"""

#CV Bootcamp code, by Lucas Khan

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import ssl
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context
# Code only runs on newer MacOS versions if this is included

#Download training data
training_data = datasets.CIFAR10(
    root='data/',
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data
test_data = datasets.CIFAR10(
    root='data/',
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    #From here, we can see that the input images have dimensions 3x32x32. We will later feed these dimensions into the input layer

    print(f"Shape of y: {y.shape} {y.dtype}")
    break


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Define a basic convolutional neural network model with one hidden layer
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*32*32,256), #Images are 3x32x32 (3 RGB channels; 32x32 dimensions)
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
#We will try three different optimizers: SGD, AdamW, and Adam
optimizers = [('SGD', torch.optim.SGD(model.parameters(), lr=1e-3)), ('AdamW', torch.optim.AdamW(model.parameters(), lr=1e-3)), ('Adam', torch.optim.Adam(model.parameters(), lr=1e-3))]

#Initialize lists for accuracies and losses per epoch so we can plot them later
accuracies = []
loss = []

#Loss function
loss_fn = nn.CrossEntropyLoss()

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device) #Send images (X) and labels (y) to neural network "device"
            pred = model(X) #Retrieve model predictions
            print(pred)
            test_loss += loss_fn(pred, y).item() #Add epoch loss to the total
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() #Counts the number of correct predictions (consistent with the label, y) and adds that number to the total
    test_loss /= num_batches #Get average loss by dividing the total loss by the number of batches
    correct /= size #Get accuracy by dividing the number of correct predictions by the total number of predictions (or sample size)
    accuracies.append(correct) #Add percent accuracy to the list of percent accuracies for use in plotting
    loss.append(test_loss) #Add average loss to a list for plotting
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 10

for optname, optimizer in optimizers: #Iterate over optimizers (SGD, AdamW, Adam)
    accuracies = [] #Empty lists for accuracies and losses with each optimizer since they each have their own line in the plot
    loss = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    plt.figure(1) #Figure 1: accuracy
    plt.plot([x for x in range(1,epochs+1)], accuracies, label=optname) #Plot the accuracy at each epoch, with the epoch on the x-axis and the accuracy on the y-axis
    plt.figure(2) #Figure 2: loss
    plt.plot([x for x in range(1,epochs+1)], loss, label=optname) #Plot the loss at each epoch, with the epoch on the x-axis and the loss on the y-axis

#Add axis labels, grid, and legend
plt.figure(1)
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.figure(2)
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.legend()
plt.show()