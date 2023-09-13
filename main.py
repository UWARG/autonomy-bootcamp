import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
from torch import flatten
from torch.nn import LogSoftmax
import torch.nn.functional as F
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

# this is our test set
test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
testset = torch.utils.data.DataLoader(test, batch_size=32,
                                      shuffle=False, num_workers=0)

# development set
train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform)
trainset = torch.utils.data.DataLoader(train, batch_size=32,
                                       shuffle=True, num_workers=0)

# this is our machine


class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()
        # here is our first convolutional layer

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        # here is our second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),



        )
        # this is our fully connected layer
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        '''
        We first create one conv layer, and then create another one (for better accuracy)
        Finally, we flatten our values up to a max of 1 and then create a fully connected layer

        '''
        x = self.conv1(x)

        x = self.conv2(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        output = self.logSoftmax(x)
        return output


# We initialize our machine neural network here
net = Net()

# We initialize some variables that we would like to plot later
cycle = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
training_loss = []
validation_loss = []
validation_accuracy = []
# Our loss function
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# Our network will have 12 run cycles
for epoch in range(12):

    '''The following variables will allow us to 
    calculate our training loss and accuracies etc'''
    run_loss = 0.0
    total_correct = 0
    total_size = 0
    train_correct = 0
    train_total = 0
    for inputs, label in trainset:
        # plug in our data to the machine and grab the output
        output = net(inputs)
        target = torch.randn(10)
        target = target.view(1, -1)
        # get network loss and run it back into our system to improve it
        loss = loss_function(output, label)

        optimizer.zero_grad()
        # used to figure out our training loss
        run_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total_correct += (predicted == label).sum().item()
        total_size += label.size(0)
        loss.backward()
        optimizer.step()
        # calculate our training accuracy
        train_correct += (torch.argmax(output, 1) == label).float().sum()
        train_total += len(label)
        # attempt to optimize weights to account for loss/gradients
    # print loss. We hope loss (a measure of wrong-ness) declines!
    print(run_loss/len(trainset))
    print("Training Accuracy: ", train_correct/train_total)
    # append our results into arrays for plotting

    training_loss.append((run_loss/len(trainset)))
    # we use these for our test set calculations
    # also resetting run_loss although another variable may have been better
    correct = 0
    total = 0
    run_loss = 0.0

    for inputs, labels in testset:
        # plug our data into our machine and retrieve uotput
        output = net(inputs)
        _, predicted = torch.max(output.data, 1)
        # calculate our correctness
        correct += (torch.argmax(output, 1) == labels).float().sum()
        total += len(labels)
    print("Accuracy: ", correct/total)
    # append our results into arrays for plotting
    validation_loss.append(1-correct/total)
    validation_accuracy.append(correct/total)
train_loss = np.array(training_loss)
validation_loss = np.array(validation_loss)

'''
The part below helps plot our graphs for training and validation loss and accuracy
We create two double-line graphs in the below area
'''
plt.plot(cycle, train_loss)
plt.plot(cycle, validation_loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Network Loss")
plt.legend(["Training Loss", "Validation Loss"])
plt.show()
plt.plot(cycle, np.array(validation_accuracy))
plt.title("Network Accuracy")
plt.show()
