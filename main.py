# Imports

from torch import flatten, optim, nn, utils, device, no_grad, max, device
from torchvision import transforms, datasets

import torch
import numpy as np
import matplotlib.pyplot as plt

class CNN(nn.Module):

    def __init__(self):

        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5, padding="same")
        self.conv2 = nn.Conv2d(12, 12, 5, padding="same")
        self.batch_norm1 = nn.BatchNorm2d(12)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(12, 24, 5, padding="same")
        self.conv4 = nn.Conv2d(24, 24, 5, padding="same")
        self.batch_norm2 = nn.BatchNorm2d(24)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(24, 36, 3, padding="same")
        self.conv6 = nn.Conv2d(36, 36, 3, padding="same")
        self.batch_norm3 = nn.BatchNorm2d(36)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(36 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 10)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = flatten(x, 1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


def fit(model, device, epochs, train_loader, val_loader):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    accuracy_sum = np.zeros(epochs)
    accuracies = np.zeros(epochs)
    losses_sum = np.zeros(epochs)
    losses = np.zeros(epochs)

    for epoch in range(epochs):

        model.train()

        # Training
        for i, data in enumerate(train_loader):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation calculations
        with no_grad():
            for i, data in enumerate(val_loader):

                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = max(outputs,dim=1)[1]

                accuracy = (predicted == labels).sum()
                accuracy_sum[epoch] += accuracy

                loss = criterion(outputs, labels)
                losses_sum[epoch] += loss

        accuracies[epoch] = (accuracy_sum[epoch] / len(val_loader.dataset))
        losses[epoch] = (losses_sum[epoch] / len(val_loader.dataset))

        print(f"epoch {epoch}: {accuracies[epoch]} accuracy, {losses[epoch]} loss")

    return losses, accuracies

def plot_results(losses, accuracies):

    plt.plot(losses, label="loss over epochs")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()

    plt.plot(accuracies, label="accuracy over epochs")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.show()

if __name__ == "__main__":

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    BATCH_SIZE = 50

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    DEVICE = device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN().to(DEVICE)
    losses, accuracies = fit(model, DEVICE, 10, train_loader, val_loader)
    plot_results(losses, accuracies)