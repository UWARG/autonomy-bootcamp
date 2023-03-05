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

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
#Source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# Your working code here

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
#transform function can be used to convert input training images to tensor, and normalize the data

batchSize = 4

trainSet = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=True, num_workers=2)

testSet = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=False, num_workers=2)
#Creates trainset from CIFAR10 dataset and apply transforms to data
#Trainloader then used to preprocess data - such as shuffling, and specify batch size as well as number of workers being used during training

#Splitting into training and testing sets. Training set used during training, and test set used to verify training progress of model

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        #Creates a 2D conv layer with 3 channel input, 6 feature maps with 5x5 kernal size
        #3 channel input used in this case as CIFAR10 dataset uses color images
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        #Three fully connected layers to reduce conv2 layer output of 16 feature maps of 5x5 filters into 10 node output
        #fc3 outputs 10 nodes which represents the target classification classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #Max pool used to take max value of 2x2 block which reduces number of inputs spatial size of input

        x = torch.flatten(x,1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        #forward function defines a single iteration of inputs through the Net architecture. 
        #ReLu function sets all is used during forward iterations to ensure the neural net will learn non-linear relationships btw input and output
        #This in turn, enables the neural net to recognize more complex relationships 

        return x

net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#Define loss criterion and optimizer for training

if __name__ == '__main__':
    for epoch in range(2):

        runningLoss = 0.0
        lossLog = []

        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            #Prepare data for GPU handling

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #Back propagating through the net

            runningLoss+= loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {runningLoss / 2000:.3f}')
                lossLog.append(runningLoss)
                runningLoss = 0.0
                #Track loss every 2000 iterations through the model 

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    #Save model weight file

    plt.figure(figsize=(10, 5))
    plt.title("Model Training Loss")
    plt.plot(lossLog, label="Training Loss")
    plt.xlabel("Iterations x 2000")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    #Create loss graph for tracking training progress