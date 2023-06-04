"""
This is a starter file to get you going. You may also include other files if you feel it's necessary.

Make sure to follow the code convention described here:
https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions

Hints:
* The internet is your friend! Don't be afraid to search for tutorials/intros/etc.
* We suggest using a convolutional neural network.
* TensorFlow Keras has the CIFAR-10 dataset as a module, so you don't need to manually download and unpack it.
"""

# Import Libraries and Modules
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms # This is a module that provides a set of functions that are used for image transformations and useful for data preprocessing
import matplotlib.pyplot as plt
import torch.nn as nn # Provides a set of classes and modules for building/training neural networks
import torch.nn.functional as F # Provides a set of functions that are commonly used in neural network operations such as Sigmoid and Hyperbolic Tangent activation functions
import torch.optim as optim

# Load and Normalize the CIFAR-10 Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Defines a sequence of transformations
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # Creates an object train_set that represents the CIFAR-10 dataset with transform applied to it and then stored in the ./data directory.
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) # Creates an object test_set that represents the CIFAR-10 dataset with transform applied to it and then stored in the ./data directory.

batch_size = 4
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

# Define Classes for Images in the Dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define a Convolutional Neural Network
class Net(nn.Module):
    def __init__(self): # models a neural network
        super().__init__() # calls the init method of parent class nn.Module
        self.conv1 = nn.Conv2d(3, 6, 5) # represents the first convolutional layer of the neural network
        self.pool = nn.MaxPool2d(2, 2) # specifies the max pooling layer
        self.conv2 = nn.Conv2d(6, 16, 5) # represents the second convolutional layer of the neural network
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # the following 3 lines define fully connected layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # applies the first convolutional layer, ReLU activation function and max pooling layer to the input x
        x = self.pool(F.relu(self.conv2(x))) # same as the line above but with the second convolutional layer
        x = torch.flatten(x, 1) # flatten all dimensions except batch, required for fully connected layers
        x = F.relu(self.fc1(x)) # the following 3 lines represent applying the previously defined fully connected layers
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x # returned x represents the predicted output of the neural network
    
net = Net() # Create Net Object

# Check if GPU is present, else use CPU for model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net.to(device)

# Define a Loss Function and Optimizer
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # Stochiastic Gradient Descent

# Train the Network
epoch_losses = []
net.train() # Puts our model in training mode
for epoch in range(20):  # Loop over the dataset more than once
    running_loss = 0.0 # variable to keep track of cumulative loss
    saved_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data # Get the inputs, data is a list of [inputs, labels]
        optimizer.zero_grad() # Zero the parameter gradients
        # Forward + backward + optimize
        outputs = net(inputs) # Get output of neural network for each input
        loss = criterion(outputs, labels) # Calculate loss by comparing accurate label value to output obtained
        loss.backward() # backpropagation
        optimizer.step() # updates network
        running_loss += loss.item() 
        if i % 2000 == 1999:  # print loss for every 2000 mini-batches
            print('%d, %5d| loss: %.3f' %(epoch+1, i+1, running_loss/2000))
            saved_loss = running_loss
            running_loss = 0.0
    epoch_losses.append(saved_loss/10000)

# Plot our training loss on a graph
epochs = range(1,21) 
plt.plot(epochs, epoch_losses, 'g', label='Training loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Test the model
total = 0
correct = 0
validation_losses = []
net.eval() # Puts our model in evaluation mode
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        validation_losses.append(loss.item())
