"""
Submission for WARG CV Subteam (Matthew Keller)

Script based on data analysis with PyTorch article: https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-analysis-intro
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

IMAGE_CLASSES = ["airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"]

BATCH_SIZE = 10
NUMBER_OF_EPOCHS = 5

class Network(nn.Module):
    """
    Convolutional Neural Network to classify images from CIFAR-10 dataset
    ...
    -------
    __init__():
        Initialize convolutional neural network
        
    forward(image: Timestamp): 
        Forward pass of image through network
    """
    def __init__(self):
        super(Network, self).__init__()
        # 3 channels represent r, g, b
        # shape: 3 features, 12 outputs, 32 x 32 image
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        # output size after conv stepp is (w-f+2P)/S + 1 = (32-5+2*1)/1 + 1 = 30
        self.bn1 = nn.BatchNorm2d(12) # Normalize inputs to layer
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=10, stride=1, padding=2)
        # (30-5+2*2)/1 + 1 = 30
        self.bn2 = nn.BatchNorm2d(24)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=10, stride=2, padding=3)
        # (30-10+2*3)/2 + 1 = 15
        self.bn4 = nn.BatchNorm2d(48)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=5, stride=2, padding=2)
        # (15-5+2*2)/2 + 1 = 8
        self.bn5 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(in_features=48*3*3, out_features=10) # Computes score for each classification in dataset


    def forward(self, input):
        # print("input shape: ", input.shape) # torch.Size([10, 3, 32, 32])
        output = F.relu(self.bn1(self.conv1(input)))     
        output = F.relu(self.bn2(self.conv2(output)))  
        output = self.pool(output)               
        output = F.relu(self.bn4(self.conv4(output)))   
        output = F.relu(self.bn5(self.conv5(output)))     
        output = output.view(-1, 48*3*3) # Flatten output to [10, 48*3*3] Tensor
        output = self.fc1(output)

        return output

def load_data():
    """
    Loads data from CIFAR-10 dataset from torchvision module

    Parameters
    ----------
    batchSize: int
        Optional parameter to specify batch size for data loader
    """

    # Transform data
    transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # mean and std found by previously loading and inspecting dataset: https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
        transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784]) 
    ])

    # Downloads CIFAR-10 dataset and loads to local memory if not already downloaded in "./data" directory
    trainSet = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transformations, download=True)
    testSet = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transformations, download=True)

    # Create data loaders for training and testing
    trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print("testLoader", testLoader.dataset)
    return trainLoader, testLoader

def set_device(model):
    """
    Sets model to be run on either CPU or GPU

    Parameters
    ----------
    model: Network
        Model to set device for
    """
    # defining device model is being run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model running on", device, "device")
    
    # Convert model to correct device
    model.to(device)

    return model, device

def test_accuracy(loader, model):
    """
    Tests accuracy of given model on given dataset by running model on dataset and computing the accuracy from all images in test set

    Parameters
    ----------
    testLoader: DataLoader
        Data set to test model on
        
    model: Network
        CNN model to test accuracy of
    """
    if loader.dataset.train:
        print("Testing model on training set")
    else:
        print("Testing model on test set")

    model.eval() # Set model to evaluation mode (rather than training)

    accuracy = 0.0
    total = 0.0
    runningLoss = 0.0

    lossFn = nn.CrossEntropyLoss() # Loss function for model

    with torch.no_grad():
        for data in loader:
            images, labels = data

            outputs = model(images)
            # label with highest energy
            _, predicted = torch.max(outputs.data, 1)

            # compute loss
            loss = lossFn(outputs,labels)
            runningLoss+=loss.item()

            # compute accuracy for given image
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    print("loader length", len(loader))

    # compute loss across all images
    testLoss=runningLoss/len(loader)

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)

    return accuracy, testLoss


def train(epoch, model, trainLoader, device):
    """
    Trains epoch of model on given dataset

    Parameters
    ----------
    epoch: int
        Epoch number to train model on
        
    model: Network
        CNN model to test accuracy of
    
    trainLoader: torchvision.DataLoader
        Data set to train model on

    device: torch.device
        Device to run model on
    """
    print('\nEpoch : %d'%epoch)
    bestAccuracy = 0.0
    model.train()
    runningLoss = 0.0

    optimizer = Adam(model.parameters())
    lossFn = nn.CrossEntropyLoss()

    for i, (data, labels) in enumerate(trainLoader, 0):
        # get the inputs
        data = Variable(data.to(device))
        labels = Variable(labels.to(device))

        optimizer.zero_grad()
        # compute network predictions and evaluate loss function
        outputs = model(data)
        
        loss = lossFn(outputs, labels)
        # backpropagate the loss
        loss.backward()

        # update weights 
        optimizer.step()

        runningLoss += loss.item()     # extract the loss value
    
    trainAccuracy, _ = test_accuracy(loader=trainLoader, model=model)
    trainLoss = runningLoss / 1000
    print("other train loss", trainLoss)


    print("Epoch has test accuracy: ", trainAccuracy)
    print("trainLoss", trainLoss)
    print("trainAccuracy", trainAccuracy)

    if trainAccuracy > bestAccuracy:
        path = "./bestAccuracy.pth"
        torch.save(model.state_dict(), path)
        bestAccuracy = trainAccuracy

    return trainAccuracy, trainLoss

def show_plots(trainAccVals, testAccVals, trainLossVals, testLossVals):
    """
    Shows graphs of accuracy and loss values for training and testing as a function of epoch

    Parameters
    ----------
    trainAccVals: list[int]
        List of training accuracy values for each epoch
    
    testAccVals: list[int]
        List of testing accuracy values for each epoch

    trainLossVals: list[int]
        List of training loss values for each epoch

    testLossVals: list[int]
        List of testing loss values for each epoch
    """

    # Accuracy
    plt.subplot(2, 1, 1)

    plt.title("Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(trainAccVals)
    plt.plot(testAccVals)
    plt.legend(['Train Accuracy', 'Test Accuracy'])

    plt.subplots_adjust(hspace=0)

    # Loss
    plt.subplot(2, 1, 2)
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(trainLossVals)
    plt.plot(testLossVals)
    plt.legend(['Train Loss', 'Test Loss'])

    plt.show()


def main():
    # Load data
    trainLoader, testLoader = load_data()

    # Initialize model 
    model = Network()

    #  Set device to run on GPU or CPU
    model, device = set_device(model)

    trainAccVals = []
    testAccVals = []

    trainLossVals = []
    testLossVals = []

    # Loop through each epoch
    for i in range(1, NUMBER_OF_EPOCHS + 1): 
        # Train model on training set
        trainAcc, trainLoss = train(epoch=i, model=model, trainLoader=trainLoader, device=device)

        # Test model on current epoch
        testAcc, testLoss = test_accuracy(loader=testLoader, model=model)

        # Add accuracy to lists to plot
        trainAccVals.append(trainAcc)
        testAccVals.append(testAcc)

        # Add loss to lists to plot
        trainLossVals.append(trainLoss)
        testLossVals.append(testLoss)

    print("trainAccVals", trainAccVals)
    print("testAccVals", testAccVals)

    print("trainLossVals", trainLossVals)
    print("testLossVals", testLossVals)

    # Plot outcomes
    show_plots(trainAccVals, testAccVals, trainLossVals, testLossVals)

    return

main()

