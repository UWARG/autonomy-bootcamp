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
import os

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import tarfile
import torch
from torch import nn
from torch import optim
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.datasets.utils import download_url

# Your working code here

def load_dataset():
    # Downloads and extracts the dataset
    # Downloading the cifar-10 dataset
    datasetUrl = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    download_url(datasetUrl, '.')

    # Extracting the dataset
    with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
        tar.extractall(path='./data')

    dataDir = './data/cifar10'
    dataset = ImageFolder(dataDir+'/train', transform=ToTensor())

    return dataset

def reserve_validation_set(dataset):
    # Separates 5000 images of the dataset to be used for validation
    # For reproducible results, randomSeed is fixed to 42
    randomSeed = 42
    torch.manual_seed(randomSeed)

    validationSize = 5000
    trainingSize = len(dataset) - validationSize

    trainingDataset, validationDataset = random_split(
        dataset, [trainingSize, validationSize]
        )
    
    batchSize = 128

    trainingDataloader = DataLoader(trainingDataset, 
                                    batchSize, 
                                    shuffle = True, 
                                    num_workers = 4, 
                                    pin_memory = True)

    validationDataloader = DataLoader(validationDataset, 
                                      batchSize*2, 
                                      num_workers = 4, 
                                      pin_memory = True)

    return trainingDataloader, validationDataloader

def init_data():
    # wrapper function
    dataset = load_dataset()
    return reserve_validation_set(dataset)

class ImageClassificationBase(nn.Module):
    """Extends nn.Module class to classify images"""
    def training_step(self, batch):
        # Passes images and labels to get loss (training)
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)

        return loss

    def validation_step(self, batch):
        # Passes images and labels to get loss and accuracy (validation)
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy

        return {'val_loss' : loss.detach(), 'val_acc' : acc}

    def validation_epoch_end(self,outputs):
        # Passes losses and accuracies to get mean accuracy/loss
        batchLosses = [x ['val_loss'] for x in outputs]
        epochLoss = torch.stack(batchLosses).mean()     # Combines losses
        batchAccs = [x ['val_acc'] for x in outputs]
        epochAcc = torch.stack(batchAccs).mean()        # Combines accuracies

        return {'val_loss': epochLoss.item(), 'val_acc': epochAcc.item()}

    def epoch_end(self, epoch, result):
        #  Prints out the results of an epoch
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
              epoch, result['train_loss'], result['val_loss'], result['val_acc']))

class Cifar10CnnModel(ImageClassificationBase):
    """ This class extends the ImageClassificationBase class to a CNN model. """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(

            # Layer 1, 2
            # Input : 3 x 32 x 32
            # Output : 64 x 16 x 16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            # Layer 3, 4
            # Input : 64 x 16 x 16
            # Output : 128 x 8 x 8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            # Layer 5, 6
            # Input : 128 x 8 x 8
            # Output : 256 x 4 x 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
         
            # Converts feature map to vectors for the 10 classes
            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, xb):
        return self.network(xb)

def accuracy(outputs, labels):
    # 
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def to_device(data, device):
    # Takes tensors and device as params, and moves tensors to the device
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """ Wrap a dataloader to move data to a device. """
    def __init__(self, dl, device):
        # Constructor
        self.dl = dl
        self.device = device

    def __iter__(self):
        # Yields a batch of data after moving it to device
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        # Returns number of batches
        return len(self.dl)




@torch.no_grad()
def evaluate(model, valLoader):
    # Freezes the layer to prevent randomization
    model.eval()
    outputs = [model.validation_step(batch) for batch in valLoader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, trainLoader, valLoader, opt_func=torch.optim.SGD):
    """
    This function trains the model and returns
    a history of training and validation 
    loss and accuracy.
    """
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in trainLoader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, valLoader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def plot_accuracy(history):
    # Plots accuracy against epochs as a percentage
    plt.title('Accuracy vs. # of Epochs')

    listOfAccuracies = []
    for epoch in history:
        listOfAccuracies.append(epoch['val_acc'])
    plt.plot(listOfAccuracies, 'x', color="black", linestyle="-")
    plt.xlabel('# of Epochs')
    plt.ylabel('Accuracy (%)')

    plt.show()

def plot_loss(history):
    # Plots loss value of the model
    plt.title('Loss vs. # of Epochs')

    trainingLosses = []
    validationLosses = []
    for epoch in history:
        trainingLosses.append(epoch['val_loss'])
        validationLosses.append(epoch['train_loss'])

    plt.plot(trainingLosses, 'x', color="red", label="Training", linestyle="-")
    plt.plot(validationLosses, 'x', color="black", label="Validaton", linestyle="-")

    plt.legend()
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss')

    plt.show()

def main():
    trainingDataloader, validationDataloader = init_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainingDataloader = DeviceDataLoader(trainingDataloader, device)
    validationDataloader = DeviceDataLoader(validationDataloader, device)
    model = to_device(Cifar10CnnModel(), device)

    history = fit(6, 0.001, model, trainingDataloader, validationDataloader, torch.optim.Adam)

    plot_accuracy(history)
    plot_loss(history)
    
if __name__ == "__main__":
    main()
