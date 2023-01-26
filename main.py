# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import device, flatten, max, nn, no_grad, optim, utils
from torchvision import transforms, datasets

class CNN(nn.Module):
    """
    Implementation of Convolutional Image Classifier.
    """
    def __init__(self):
        """
        Layer Definitions.
        """
        super().__init__()

        # Convolution Layers
        self.conv1 =nn.Conv2d(3, 24, 5, padding="same")
        self.conv2 =nn.Conv2d(24, 24, 5, padding="same")
        self.conv3 =nn.Conv2d(24, 48, 3, padding="same")
        self.conv4 =nn.Conv2d(48, 48, 3, padding="same")
        self.conv5 =nn.Conv2d(48, 96, 3, padding="same")
        self.conv6 =nn.Conv2d(96, 96, 3, padding="same")

        # Batch Normalization Layers
        self.batch_norm1 =nn.BatchNorm2d(24)
        self.batch_norm2 =nn.BatchNorm2d(48)
        self.batch_norm3 =nn.BatchNorm2d(96)

        # Pooling Layers
        self.pool1 =nn.MaxPool2d(2, 2)
        self.pool2 =nn.MaxPool2d(2, 2)
        self.pool3 =nn.MaxPool2d(2, 2)

        # Global Max Pooling Layer
        self.max_pool =nn.AdaptiveMaxPool2d(1)
        self.flatten =nn.Flatten()

        # Fully Connected Layers
        self.fc1 =nn.Linear(96, 256)
        self.fc2 =nn.Linear(256, 512)
        self.fc3 =nn.Linear(512, 10)

        # Dropout and Relu Layers
        self.dropout1 =nn.Dropout(0.2)
        self.dropout2 =nn.Dropout(0.2)
        self.relu =nn.ReLU()

    def forward(self, x):
        """
        Model ARchitecture.
        """
        # Convolution layers before first downsampling
        x =self.conv1(x)
        x =self.batch_norm1(x)
        x =self.relu(x)
        x =self.conv2(x)
        x =self.batch_norm1(x)
        x =self.relu(x)
        x =self.pool1(x)

        # Convolution layers before second downsampling
        x =self.conv3(x)
        x =self.batch_norm2(x)
        x =self.relu(x)
        x =self.conv4(x)
        x =self.batch_norm2(x)
        x =self.relu(x)
        x =self.pool2(x)

        # Convolution layers before third downsampling
        x =self.conv5(x)
        x =self.batch_norm3(x)
        x =self.relu(x)
        x =self.conv6(x)
        x =self.batch_norm3(x)
        x =self.relu(x)
        x =self.pool3(x)

        # Global Max Pooling
        x =self.max_pool(x)
        x =self.flatten(x)

        # Fully Connected Layers
        x =self.fc1(x)
        x =self.relu(x)
        x =self.dropout1(x)
        x =self.fc2(x)
        x =self.relu(x)
        x =self.dropout2(x)
        x =self.fc3(x)

        return x


def fit(model, device, epochs, train_loader, val_loader, criterion, optimizer):
    """
    Method to train model
    """
    # Arrays to track statistics
    accuracies =np.zeros(epochs)
    losses =np.zeros(epochs)
    accuracies_val =np.zeros(epochs)
    losses_val =np.zeros(epochs)

    # Epoch loop
    for epoch in range(epochs):

        # Enables training
        model.train()

        # Training step
        for i, data in enumerate(train_loader):

            inputs, labels =data
            inputs, labels =inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs =model(inputs)
            loss =criterion(outputs, labels)
            # Back Propogation
            loss.backward()
            optimizer.step()

        # Generate loss and accuracy statistics
        with no_grad():

            # Calculate sum of training loss and accuracy
            for i, data in enumerate(train_loader):

                # Get prediction on training data
                inputs, labels =data
                inputs, labels =inputs.to(device), labels.to(device)
                outputs =model(inputs)
                predicted =max(outputs,dim=1)[1]

                # Get metrics
                accuracy =(predicted ==labels).sum()
                accuracies[epoch] +=accuracy
                loss = criterion(outputs, labels)
                losses[epoch] +=loss

            # Calculate sum of validation loss and accuracy
            for i, data in enumerate(val_loader):

                # Get prediction on validation data
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs =model(inputs)
                predicted =max(outputs,dim=1)[1]

                # Get metrics
                accuracy =(predicted ==labels).sum()
                accuracies_val[epoch] +=accuracy
                loss =criterion(outputs, labels)
                losses_val[epoch] +=loss

        # Calculate training loss and accuracy from sum
        len_train_data =len(train_loader.dataset)
        accuracies[epoch] /=len_train_data
        losses[epoch] /=len_train_data

        # Calculate validation loss and accuracy from sum
        len_val_data =len(val_loader.dataset)
        accuracies_val[epoch] /=len_val_data
        losses_val[epoch] /=len_val_data

        # Print current epoch stats
        print(f"epoch {epoch}. acc: {accuracies[epoch]}, val_acc: \
            {accuracies_val[epoch]}, loss: {losses[epoch]}, val_loss: {losses_val[epoch]}")

    return losses, accuracies, losses_val, accuracies_val

def plot_results(losses, accuracies, losses_val, accuracies_val):
    """
    Plot metrics for training and validation
    """
    # Plot loss
    plt.plot(losses, label="loss")
    plt.plot(losses_val, label="validation loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

    # Plot accuracy
    plt.plot(accuracies, label="accuracy")
    plt.plot(accuracies_val, label="validation accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    # Load Data
    BATCH_SIZE = 128
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Instantiate model
    DEVICE = device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train model
    losses, accuracies, losses_val, accuracies_val = fit(model, DEVICE, 10, train_loader, val_loader, criterion, optimizer)
    plot_results(losses, accuracies, losses_val, accuracies_val)
