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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader

class Model(nn.Module):
    """
    This class implements a Image Classification model
    """
    def __init__(self):
        super().__init__()
        # Convolution Layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1)
        # Maxpooling Layer
        self.pool = nn.MaxPool2d(2, 2)
        # Fully Connected Layers
        self.linear1 = nn.Linear(1024, 256)
        self.linear2 = nn.Linear(256, 10)
        # Dropout Layer
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # BATCHSIZE*3*32*32 -> BATCHSIZE*32*16*16
        x = self.pool(F.relu(self.conv2(x))) # BATCHSIZE*32*16*16 -> BATCHSIZE*64*8*8
        x = self.pool(F.relu(self.conv3(x))) # BATCHSIZE*64*8*8 -> BATCHSIZE*128*4*4
        x = self.dropout(x) # Drops out 30% of the Neurons
        x = x.view(-1, 64*4*4) # Flatten the data
        
        x = F.relu(self.linear1(x)) # BATCHSIZE*1024 -> BATCHSIZE*256
        x = F.log_softmax(self.linear2(x), dim = 1) # BATCHSIZE*256 -> BATCHSIZE*10
        return x

def train(model, device, train_loader, val_loader, epochs, criterion = nn.CrossEntropyLoss()):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) 
    accuracies = np.zeros(epochs)
    losses = np.zeros(epochs)
    model.train()
    for epoch in range(epochs):
        for id, (datas, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(datas)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            for id, (datas, labels) in enumerate(val_loader):
                outputs = model(datas)
                predicted = torch.max(outputs,dim=1)[1]
                accuracies[epoch] += (predicted == labels).sum()
                losses[epoch] += criterion(outputs, labels)
        accuracies[epoch] /= len(val_loader.dataset)/100
        losses[epoch] /= len(val_loader)
        print(f"EPOCH {epoch}: Loss: {losses[epoch]:.4f} Accuracy: {accuracies[epoch]}")
    return accuracies

if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 10
    PATH = './cifar_net'

    # Fetching and transforming datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    trainset = datasets.CIFAR10(root = './CIFAR10', train=True, download=True, transform=transform) # fetch the train datasets
    testset = datasets.CIFAR10(root = './CIFAR10', train=False, download=True, transform=transform) # fetch the test datasets
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 8, pin_memory = True)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 8, pin_memory = True)
    model = Model().to(DEVICE)
    accuracies = train(model, DEVICE, train_loader, test_loader, EPOCHS)
    plt.plot(accuracies)
    plt.xlabel("EPOCH")
    plt.ylabel("Accuracy(%)")
    plt.xticks(np.arange(0, 10, 1))
    plt.title("Validation Accuracy")
    plt.show()