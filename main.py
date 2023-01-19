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

# Your working code here
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
min_valid_loss = np.inf
batch_size = 8
x_coords = []  #storing the x-values of graph
y_train = []  #storing y-values of the training set
y_valid = []  #storing y-values of the validation set

#downloading the sets, and splitting the training set into a smaller training set and a validation set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainset, validset = random_split(trainset,[45000,5000])
trainloader = DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

validloader = DataLoader(validset, batch_size=batch_size, shuffle=False)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# use class method to make neural network with 2d convolutions, max pooling, then linear transformations to reduce channel amount to 10
class Net(nn.Module): 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 128, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 1, 1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(9216, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x): #layer order
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#looping through the dataset for 15 epochs
for epoch in range(15):  
    x_coords.append(epoch+1)
    running_loss = 0.0

    #training with the training set
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    #using validation set to check model efficacy, if the validation loss is lower, the current model is saved to cifar_net.pth
    for data, labels in validloader:
        target = net(data)
        loss = criterion(target,labels)
        valid_loss = loss.item() * data.size(0)

    #printing out training loss and validation loss per epoch
    print(f'Epoch {epoch+1} \t\t Training Loss: {running_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(validloader)}')
    #append point values to be used in the graph
    y_train.append((running_loss / len(trainloader)))
    y_valid.append((valid_loss / len(validloader)))

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        #saving the model if it is better than the previous one
        torch.save(net.state_dict(), "./cifar_net.pth")

print('Finished Training')


correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

#testing on final test set
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

#some code to make the matplotlib graph
plt.plot(x_coords, y_train, label = "Training Loss")
plt.plot(x_coords, y_valid, label = "Validation Loss")
plt.xlabel("Epoch Number", fontsize=15)
plt.ylabel("Losses", fontsize=15)
plt.legend()
plt.show()