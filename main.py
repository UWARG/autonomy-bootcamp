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
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


#device configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper-parameters
num_epochs = 4
batch_size = 2
learning_rate = 0.001


#dataset has PILImage images of range [0,1]
#You can transform them to tensors of normalized range [-1, 1]
tensor_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])


# this represents the data that the module with be trained on
training_data = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = tensor_transform)


# This represents the data set that is going to be used to evaluate the trainned module
evaluation_data = torchvision.datasets.CIFAR10(root = './data', train= False, download = True, transform = tensor_transform)


# this takes in each dataset (train and evaluation and sets a batch size)
# shuffle is set to true for train as if the module is trained on the same module it woould mostly like be in accurate 
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size = batch_size, shuffle = True)
Evaluation_dataloader = torch.utils.data.DataLoader(evaluation_data, batch_size = batch_size, shuffle = False)

# this is an object containing all the different objects contained in the dataset
different_objects_in_data = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#implement conv net
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.network = nn.Sequential(
            # first layer input size of 3 output size of 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # relu used as avtivation function
            nn.ReLU(),
            #second layer with input size of 32(same as output size of first layer) and an out put size of 64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4
            nn.BatchNorm2d(256),

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
           
    def forward(self, xb):
        return self.network(xb)
  
model = ImageClassifier().to(device)
# soft max is included here 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

accuracyList = []
loss_values = []

n_total_steps = len(training_dataloader)
for epoch in range(num_epochs):
    n_correct = 0
    n_samples = 0
    for i, (images, label) in enumerate(training_dataloader):
        #origin shape: [4,3,32,32] = 4,3, 1024
        #input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = label.to(device)

        #forward pass
        output = model(images)
        _, predicted = torch.max(output, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        loss = criterion(output, labels)

        #backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if (i+1) % 2000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/ {n_total_steps}], Loss: {loss.item():.4f}')
    loss_values.append(loss.item())
    accuracyPerEpoch = 100*( n_correct/n_samples)
    accuracyList.append(accuracyPerEpoch)

print('Finished Training')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range (10)]
    n_class_samples = [0 for i in range (10)]
    for images, labels in Evaluation_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        accuracy = torch.tensor(torch.sum(predicted == labels).item() / len(predicted))

        _, predicted = torch.max(output, 1)

        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()   

    accuracy1 = 100 * (n_correct / n_samples)
    print(f'Accuracy of the network: {accuracy1}%')

num_epochs = [1,2,3,4]

plt.plot(num_epochs, accuracyList)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
    
plt.plot(num_epochs, loss_values)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()








