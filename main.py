"""
This is a starter file to get you going. You may also include other files if you feel it's necessary.

Make sure to follow the code convention described here:
https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions

Hints:
* The internet is your friend! Don't be afraid to search for tutorials/intros/etc.
* We suggest using a convolutional neural network.
* TensorFlow Keras has the CIFAR-10 dataset as a module, so you don't need to manually download and unpack it.
"""
"""
Credits
Chatgpt for the model design and 
help with the structure of the functions
"""
# Import whatever libraries/modules you need

import numpy as np

# Your working code here
# Help received from chat gpt and the pytorch tutorial
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


# Prepare the data!
def prepare_data(train_data, val_data, test_data, batch_size, num_workers):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# Define the neural network architecture
def create_model():
  class CIFARNet(nn.Module):
      def __init__(self):
          super(CIFARNet, self).__init__()
          
          self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
          self.bn1 = nn.BatchNorm2d(32)
          self.relu1 = nn.ReLU(inplace=True)
          self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
          self.bn2 = nn.BatchNorm2d(32)
          self.relu2 = nn.ReLU(inplace=True)
          self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
          
          self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
          self.bn3 = nn.BatchNorm2d(64)
          self.relu3 = nn.ReLU(inplace=True)
          self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
          self.bn4 = nn.BatchNorm2d(64)
          self.relu4 = nn.ReLU(inplace=True)
          self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
          
          self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
          self.bn5 = nn.BatchNorm2d(128)
          self.relu5 = nn.ReLU(inplace=True)
          self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
          self.bn6 = nn.BatchNorm2d(128)
          self.relu6 = nn.ReLU(inplace=True)
          self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
          
          self.fc1 = nn.Linear(in_features=128*4*4, out_features=512)
          self.bn7 = nn.BatchNorm1d(512)
          self.relu7 = nn.ReLU(inplace=True)
          self.dropout = nn.Dropout(p=0.5)
          
          self.fc2 = nn.Linear(in_features=512, out_features=10)
          self.softmax = nn.Softmax(dim=1)
          
      def forward(self, x):
          x = self.conv1(x)
          x = self.bn1(x)
          x = self.relu1(x)
          x = self.conv2(x)
          x = self.bn2(x)
          x = self.relu2(x)
          x = self.pool1(x)
          
          x = self.conv3(x)
          x = self.bn3(x)
          x = self.relu3(x)
          x = self.conv4(x)
          x = self.bn4(x)
          x = self.relu4(x)
          x = self.pool2(x)
          
          x = self.conv5(x)
          x = self.bn5(x)
          x = self.relu5(x)
          x = self.conv6(x)
          x = self.bn6(x)
          x = self.relu6(x)
          x = self.pool3(x)
          
          x = x.view(-1, 128*4*4)
          x = self.fc1(x)
          x = self.bn7(x)
          x = self.relu7(x)
          x = self.dropout(x)
          
          x = self.fc2(x)
          x = self.softmax(x)
          
          return x

  net = CIFARNet()
  return net

# Train function
def train_model(model, train_loader, criterion, optimizer, device):
  model.train()
  for inputs, targets in train_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()
  return loss.item()
# Evaluate model
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    return [loss.item(), accuracy]

# Define ploting function
def plot_loss(train_loss, val_loss):
  # create an array of indices to use as the x-axis
  num_epochs = len(val_loss, )
  indices = range(1, num_epochs + 1)

  # plot the validation losses in blue
  plt.plot(indices, val_loss, 'b', label='Validation Loss')

  # plot the training losses in red
  plt.plot(indices, train_loss, 'r', label='Training Loss')

  # set axis labels and title
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training and Validation Losses')

  # add a legend
  plt.legend()

  # display the plot
  plt.show()


def main():
    # 1. Data preprocessing
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    data_dir = './data'

    download = False
     # Check if the CIFAR-10 data files exist
    if not os.path.isdir(data_dir):
        download = True
        print("CIFAR-10 dataset not found in the data directory!")
    train_data = trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                download=download, transform=transform)
    
    val_data =  torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                          download=download, transform=transform)
    
    test_data = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                          download=download, transform=transform)
     
    batch_size = 4

    num_workers = 2
    
    train_loader, val_loader, test_loader = prepare_data(train_data, val_data, test_data, batch_size, num_workers)
    
    # 2. Model architecture
    model = create_model()
    
    # 3. Compile the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # 4. Train and evaluate the model
    NUMBER_OF_EPOCHS = 20
    train_loss=[]
    val_loss=[]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(NUMBER_OF_EPOCHS):
        train_loss.append(train_model(model, train_loader, criterion, optimizer, device))
        val_loss_accuracy = evaluate_model(model, val_loader, criterion, device)
        val_loss.append(val_loss_accuracy[0])
        print(f"Epoch {epoch+1}, Training loss: {train_loss[epoch]},Validation loss: {val_loss_accuracy[0]}, Validation accuracy: {val_loss_accuracy[1]:.2f}%")
    
    torch.save(model.state_dict(), 'my_model.pth')

    
    # 5. Plot the train and val loss over epochs
    plot_loss(train_loss, val_loss)


    # 6. Get the final accuracy
    val_loss_accuracy = evaluate_model(model, val_loader, criterion, device)
    print(f"Validation accuracy: {val_loss_accuracy[1]:.2f}%")


    # 7. Use the model for predictions
   # with torch.no_grad():
   #     for inputs, _ in test_loader:
   #         inputs = inputs.to(device)
   #         outputs = model(inputs)
   #         _, predicted = torch.max(outputs.data, 1)
   #         # do something with the predicted labels

if __name__ == '__main__':
    main()
