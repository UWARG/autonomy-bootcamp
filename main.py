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
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
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
               # first convolution layer
    #     self.conv1 = nn.Conv2d(3, 32, 3)
    #     self.relu = nn.ReLU(),
    #     # second convolution layer
    #     self.conv2 =  nn.Conv2d(32, 64, 3)     
    #     self.relu = nn.ReLU()  
    #     # third convolution layer
    #     self.conv3 = nn.Conv2d(16, 32, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.fc1 = nn.Linear(32 * 5 * 5, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 10)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     # added the third convolution layer here
    #     x = self.pool(F.relu(self.conv3(x)))
    #     x = x.view(-1, 32*5*5)
    #     # x = torch.flatten(x,1)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)

    #     return x
    #     # this represent the first convolution layer
    #     # 3 is the input size, 6 is the output size and the kernel size is 5 
    #     self.conv1 = nn.Conv2d(3, 6, 5)
    #     # this represents the pooling layer 
    #     #kernel size of 2 and stride of 2 
    #     self.pool = nn.AvgPool2d(2, 2)
    #     #second convolution later
    #     # hence the input should be equalt to the size of the output of the first layer 
    #     self.conv2 =  nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 10)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 16*5*5)
    #     x = torch.flatten(x,1)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)

    #     return x


model = ImageClassifier().to(device)
# soft max is included here 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_values = []

n_total_steps = len(training_dataloader)
for epoch in range(num_epochs):
    for i, (images, label) in enumerate(training_dataloader):
        #origin shape: [4,3,32,32] = 4,3, 1024
        #input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = label.to(device)

        #forward pass
        output = model(images)
        loss = criterion(output, labels)

        loss_values.append(loss.item())


        #backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if (i+1) % 2000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/ {n_total_steps}], Loss: {loss.item():.4f}')

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

        # max returns (value, index)
        _, predicted = torch.max(output, 1)


        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        # for i in range(batch_size):
        #     label = labels[i]
        #     pred = predicted[i]

        #     if (label == pred):
        #         n_class_correct[label] += 1
        #     n_class_samples[label] += 1

    accuracy1 = 100 * (n_correct / n_samples)
    print(f'Accuracy of the network: {accuracy1}%')


   

total_iterations = len(loss_values)
iterations_per_epoch = len(training_dataloader)
num_epochs = total_iterations / iterations_per_epoch
plt.plot(loss_values, '-rx')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()



