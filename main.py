"""
This is a starter file to get you going. You may also include other files if you feel it's necessary.

Make sure to follow the code convention described here:
https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions

Hints:
* The internet is your friend! Don't be afraid to search for tutorials/intros/etc.
* We suggest using a convolutional neural network.
* TensorFlow Keras has the CIFAR-10 dataset as a module, so you don't need to manually download and unpack it.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

# setting model parameters
NUM_EPOCHS = 10
opt_func = torch.optim.Adam
LR = 0.001
BATCH_SIZE = 128

# getting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# setting transforms
transform_train = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# loading data
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform_train)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def accuracy(output, label):
    # checking for the accuracy between labels(target) and what the model outputs
    output_index = torch.max(output, 1)[1]
    total_accurate = (output_index == label).sum()
    return total_accurate.item() / len(label)


class ImageClassifier(nn.Module):
    # model
    def __init__(self):
        super().__init__()
        self.neural_network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, image):
        return self.neural_network(image)

    def step(self, batch, testing=False):
        # going through the batch
        batch_images, batch_labels = batch[0].to(device), batch[1].to(device)
        result = self(batch_images)
        loss = F.cross_entropy(result, batch_labels)
        if testing:
            # if testing give some more information
            acc = accuracy(result, batch_labels)
            return loss.detach().item(), acc
        return loss

    @staticmethod
    def validation_epoch_end(testing_loss, testing_accuracy):
        # getting the mean of all the loss and accuracy
        epoch_loss = np.mean(testing_loss)
        epoch_accuracy = np.mean(testing_accuracy)
        return epoch_loss, epoch_accuracy

    @staticmethod
    def epoch_end(epoch, epoch_train_loss, epoch_test_loss, epoch_accuracy):
        # print information at the end of each epoch
        print(f"Epoch [{epoch}], train_loss: {epoch_train_loss:.4f},"
              f" val_loss: {epoch_test_loss:.4f}, val_acc: {epoch_accuracy:.4f}")


@torch.no_grad()
def evaluate(model, test_loader):
    # testing the model to see how accurate it is
    model.eval()
    testing_output = [model.step(batch, testing=True) for batch in test_loader]
    loss = [x[0] for x in testing_output]
    testing_accuracy = [x[1] for x in testing_output]
    return model.validation_epoch_end(loss, testing_accuracy)


def fit(epochs, learning_rate, classifier, train_loader, test_loader, optimization_func):
    test_loss = []
    test_accuracy = []
    train_loss = []
    optimizer = optimization_func(classifier.parameters(), learning_rate)
    for epoch in range(epochs):
        # Training Phase
        classifier.train()
        losses = []
        for batch in train_loader:
            loss = classifier.step(batch)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # testing phase
        result = evaluate(classifier, test_loader)
        test_loss.append(result[0])
        test_accuracy.append(result[1])
        train_loss.append(np.mean(losses))
        classifier.epoch_end(epoch, np.mean(losses), result[0], result[1])

    return test_loss, test_accuracy, train_loss


model = ImageClassifier().to(device)
training_data = fit(NUM_EPOCHS, LR, model, train_loader, test_loader, opt_func)


def plot_losses(training_data):
    # plotting the data
    training_loss = training_data[2]
    testing_loss = training_data[0]
    index = [x for x in range(len(training_loss))]
    ax1 = plt.subplot()
    l1, = ax1.plot(index, training_loss, color='red')
    ax1.set_ylabel("training loss")
    ax2 = ax1.twinx()
    l2, = ax2.plot(index, testing_loss, color='orange')
    ax2.set_ylabel("testing loss")
    plt.legend([l1, l2], ["training loss", "validation loss"])
    plt.show()


plot_losses(training_data)
