# UWARG Computer Vision Bootcamp

from time import time
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 20
BATCH_SIZE = 1024


class VGG11(nn.Module):
    def __init__(self) -> None:
        super(VGG11, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # have to transpose for linear layers
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


if __name__ == "__main__":
    root = os.path.dirname(__file__)
    dataset_path = os.path.join(root, "datasets")

    training_data = CIFAR10(
        root=dataset_path, train=True, transform=T.ToTensor(), download=True
    )
    test_data = CIFAR10(
        root=dataset_path, train=False, transform=T.ToTensor(), download=True
    )

    model = VGG11()
    model.to(DEVICE)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4)

    loss_func = nn.CrossEntropyLoss()

    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    # Store loss and accuracy values at each epoch
    training_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(NUM_EPOCHS):
        print(f"[epoch: {epoch}]")

        model.train()  # sets model to training mode
        t = time()
        batch_loss = 0
        train_conf_matrix = np.zeros((10, 10), dtype=np.int32)
        for data, targets in train_loader:
            # Move data and targets to the same device as the model
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            # Zero-out gradients - torch accumulates gradients by default,
            # so we have to manually reset them in each iteration
            optimizer.zero_grad()

            # Run forward pass - model returns the logits with shape (batch_size, num_classes)
            logits = model(data)

            # Compute loss
            loss = loss_func(logits, targets)

            # Compute accuracy
            pred_classes = torch.argmax(logits, dim=1)

            # Update confusion matrix
            for target, pred in zip(targets, pred_classes):
                train_conf_matrix[target, pred] += 1

            # Perform backpropagation and update model parameters
            loss.backward()
            optimizer.step()

            # Store loss
            batch_loss += loss.item()

        # Loss should be averaged over batches in an epoch
        loss = batch_loss / len(train_loader)
        training_losses.append(loss)

        # Calculate accuracy, sum of diagonal of confusion matrix is number of correct predictions
        train_acc = np.diag(train_conf_matrix).sum() / train_conf_matrix.sum() * 100
        train_accuracies.append(train_acc)

        print(f"Training time: {time() - t:.2f}s")

        model.eval()  # sets model to evaluation mode
        t = time()
        batch_loss = 0
        test_conf_matrix = np.zeros((10, 10), dtype=np.int32)
        with torch.no_grad():
            for data, targets in test_loader:
                # Make sure data is on the same device as the model
                data = data.to(DEVICE)
                targets = targets.to(DEVICE)

                # Run forward pass
                logits = model(data)

                # Compute loss
                loss = loss_func(logits, targets)
                batch_loss += loss.item()

                # Get class predictions
                pred_classes = torch.argmax(logits, dim=1)

                # Update confusion matrix
                for target, pred in zip(targets, pred_classes):
                    test_conf_matrix[target, pred] += 1

        loss = batch_loss / len(test_loader)
        test_losses.append(loss)

        test_acc = np.diag(test_conf_matrix).sum() / test_conf_matrix.sum() * 100
        test_accuracies.append(test_acc)

        print(f"Testing time: {time() - t:.2f}s")

    plt.plot(training_losses, "r-", label="Training losses")
    plt.plot(test_losses, "k-", label="Test losses")
    plt.xticks(np.arange(NUM_EPOCHS + 1, step=5))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.plot(train_accuracies, "b-", label="Training accuracy")
    plt.plot(test_accuracies, "g-", label="Test accuracy")
    plt.xticks(np.arange(NUM_EPOCHS + 1, step=5))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
