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

import numpy as np

# Your working code here
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
import torch # Importing Libraries and Loading CIFAR-10 Dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Model Building
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Add padding
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # Add padding
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # Add padding and New layer
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)  # Add padding and New layer
        self.fc1 = nn.Linear(256 * 2 * 2, 256)  # Adjust input size accordingly
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)  # New layer
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)  # New layer
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Initialize your network and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Load model if it exists
if os.path.exists('model_state_dict.pth'):
    print("Loading existing model")
    net.load_state_dict(torch.load('model_state_dict.pth'))
    net.train()


# Data transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)


# Exploring the Dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Function to show images
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

    # Get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)  # Use Python's built-in next() function here



# Model Training
print("Training")

train_losses = []
val_losses = []
accuracies = []

for epoch in range(6):  # 
    running_train_loss = 0.0
    running_val_loss = 0.0
    
    # Training loop
    net.train()  # Set the model to training mode
    for i, data in enumerate(trainloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item()
    
    # Validation loop
    correct = 0
    total = 0
    net.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation to speed up computations
        for i, data in enumerate(testloader):
            inputs, labels = data
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


            running_val_loss += loss.item()

    # Calculate and store accuracies
    epoch_accuracy = 100 * correct / total  
    accuracies.append(epoch_accuracy)  
    
    # Calculate and store average losses
    avg_train_loss = running_train_loss / len(trainloader)
    avg_val_loss = running_val_loss / len(testloader)
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Accuracy: {epoch_accuracy:.2f}%")
            

# save the model state dictionary after training
torch.save(net.state_dict(), 'model_state_dict.pth')

# Plotting both the curves simultaneously
plt.figure()  # Create a new figure
plt.plot(train_losses, color='green', label='Training Loss')
plt.plot(val_losses, color='blue', label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()
plt.show()

# Plotting the accuracy curve
plt.figure()  # Create a new figure
plt.plot(accuracies, color='red', label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Epochs")
plt.legend()
plt.show()