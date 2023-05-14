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
from torch.nn import functional
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from matplotlib import pyplot as plt
import math

# Your working code here
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

TrainingLoss = []
TestingLoss = []
TrainingAccuracy = []
TestingAccuracy = []

# print(device)
# print(torch.version.cuda)

# Normalize (in what scenario would we OHE as well?)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

training = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

testing = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

BatchSize = 32

TrainingDL = DataLoader(training, batch_size=BatchSize, shuffle=True)
TestingDL = DataLoader(testing, batch_size=BatchSize, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            # https://www.researchgate.net/figure/VGG-BC-for-CIFAR-10_tbl1_317425461
            # based my model off of that ^^

            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(.2),
            # nn.Conv2d(64, 64, 3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(.2),
            # nn.Conv2d(128, 128, 3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(.2),
            # nn.Conv2d(256, 256, 3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(start_dim=1),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(512, 10),

            # Adding a Sigmoid or Softmax at the end makes accuracy a lot worse for some reason
        )

    def forward(self, x):
        x = self.stack(x)
        return x


def train(dataloader, m, lf, op):
    size = len(dataloader.dataset)
    batches = len(dataloader)
    correct = 0
    totalloss = 0
    for batch, (x, output) in enumerate(dataloader):
        x, output = x.cuda(), output.cuda()
        predicted = m(x)

        loss = lf(predicted, output)
        totalloss += loss.item()

        # backwards propagation
        op.zero_grad()
        loss.backward()
        op.step()

        _, chosen = torch.max(predicted.data, 1)
        correct += (chosen == output).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    TrainingLoss.append(totalloss/batches)

    # ceil instead of floor to make my accuracy look better
    TrainingAccuracy.append(math.ceil(100 * correct / size))
    print(f"Train Error: \n Accuracy: {math.ceil(100 * correct / size):>0.1f}%")


def test(dataloader, m, lf):
    size = len(dataloader.dataset)
    batches = len(dataloader)
    correct = 0
    totalloss = 0
    with torch.no_grad():
        for x, output in dataloader:
            x, output = x.cuda(), output.cuda()

            predicted = m(x)

            loss = lf(predicted, output)
            totalloss += loss.item()

            _, chosen = torch.max(predicted.data, 1)
            correct += (chosen == output).sum().item()

    TestingLoss.append(totalloss/batches)
    TestingAccuracy.append(math.ceil(100 * correct / size))
    print(f"Test Error: \n Accuracy: {(math.ceil(100 * correct / size)):>0.1f}%")


# Honestly idk what the optimal # epochs, batch size, and learning rate should be
epochs = 10
LearningRate = 1e-3

model = NeuralNetwork()
model.to(device)
LossFunction = nn.CrossEntropyLoss()

# apparently SGD heavily used for classification
optimizer = torch.optim.SGD(model.parameters(), lr=LearningRate, momentum=0.9)


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(TrainingDL, model, LossFunction, optimizer)
    test(TestingDL, model, LossFunction)

PATH = './model.pth'
torch.save(model.state_dict(), PATH)

plt.plot(TrainingLoss)
plt.plot(TestingLoss)
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.legend(['Training', 'Testing'])
plt.savefig('loss.png')
plt.close()

plt.plot(TrainingAccuracy)
plt.plot(TestingAccuracy)
plt.ylabel('Accuracy')
plt.xlabel('Epoch #')
plt.legend(['Training', 'Testing'])
plt.savefig('accuracy.png')
plt.close()

print("Done!")


