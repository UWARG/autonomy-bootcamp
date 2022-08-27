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
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np 
import torch.optim as optim
from tqdm import tqdm


# Your working code here

"""
Function: gpu_check()

Checks if cuda is available to train on GPU

Paramteres: 
None

Return: 
Returns True or False statements based on if cuda is available and training on the GPU is possible.

"""

# Device configuration
def gpu_check():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('GPU state:', device)
    return device

# transfroms.Compose([]) allows you to speicfy a list of transformations applied on the data
transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

"""
Function: load_data()

Load data from torchvision and process

Parameters: 

None

Return: 
Returns both the training/testing dataset after normalization 

"""
# Load and Process Data
def load_data():
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                            download=True, transform=transform)

    # Load the selected dataset for training, specifying number of samples per batch, shuffle, and number of proccesses
    trainloader = torch.utils.data.DataLoader(train_set,batch_size=32,
                                                shuffle=True,num_workers=0)

    test_set = torchvision.datasets.CIFAR10('data', train=False, 
                                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test_set,batch_size=32,
                                            shuffle=False,num_workers=0)
    return trainloader, testloader

# Data classes for training and testing
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
print(classes)

"""
Function: cnn()
Implement Convolutional Neural Network using Sequential method

Parameters: 
device

Returns:
model

"""

def cnn(device):
    model = nn.Sequential(
        nn.Conv2d(3,32, kernel_size=3, padding=1),          # Feature extraction
        nn.ReLU(),                                          # Applies non-linearity
        nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),                                  # Downsamples the feature map
        nn.BatchNorm2d(64),                                 # Batch Normalization layer
        
        # Increasing the depth to 64
        nn.Conv2d(64,128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2), 
        nn.BatchNorm2d(128),
        
        # Increasing the depth to 128
        nn.Conv2d(128,256, kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
       
       # Flattening the model 
        nn.Flatten(),                                        # Flattens input by reshaping it into a one-dimensional tensor
        nn.Linear(256*4*4, 1024),                            # Applies a linear transformation to the incoming data
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10))
    
    # Send model to device
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    print(model)
    return model

"""
Function: loss_opt
Assigns a loss function by compiling the model with cross-entropy and optimization

Parameters: 
model

Returns:
lossFunction, optimizer

"""
# Compiling model with cross-entropy and optimizing
def loss_opt(model):
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return lossFunction, optimizer


# Setting model training Loss and Accuracy variables 
train_losses=[]
train_accu=[]

"""
Function: train()
Trains the model

Paramters:
epoch, model, trainloader, device, lossFunction, optimizer

Returns:
None

"""

# Train the model
def train(epoch, model, trainloader, device, lossFunction, optimizer):
  print('\nEpoch : %d'%epoch)
  
  model.train()

  running_loss=0
  correct=0
  total=0


  # Determining training loss from model predictions
  for data in tqdm(trainloader):
    
    inputs,labels=data[0].to(device),data[1].to(device)
    outputs=model(inputs)
    loss=lossFunction(outputs,labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    
    _, predicted = outputs.max(1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()

  # Defining training loss
  train_loss=running_loss/len(trainloader)
  accu=100.*correct/total

  # Printing model 
  train_accu.append(accu)
  train_losses.append(train_loss)
  print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))

# Setting model validation(test) Loss and Accuracy variables 
eval_losses=[]
eval_accu=[]


"""
Function: test()
Testing the model

Parameters:
epoch, model, testloader, device, lossFunction

Returns:
None

"""
# Testing the model
def test(epoch, model, testloader, device, lossFunction):
  model.eval()

  running_loss=0
  correct=0
  total=0

  with torch.no_grad():
    for data in tqdm(testloader):
      images,labels=data[0].to(device),data[1].to(device)
      
      outputs=model(images)

      loss= lossFunction(outputs,labels)
      running_loss+=loss.item()
      
      _, predicted = outputs.max(1)
      total += labels.size(0)
      correct += predicted.eq(labels).sum().item()
  
  test_loss=running_loss/len(testloader)
  accu=100.*correct/total

  eval_losses.append(test_loss)
  eval_accu.append(accu)

  # Printing model testing progress 
  print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))


"""
Function: train_model()
Runs the train and test functions based on the epochs specified

Parameters:
model, trainloader, device, lossFunction, optimizer

Returns:
None

"""
# Runs train_model
def train_model(model, trainloader, device, lossFunction, optimizer):
    epochs=10
    for epoch in range(1,epochs+1): 
        train(model, trainloader, device, lossFunction, optimizer)
        test(model, trainloader, device, lossFunction, optimizer)


"""
Function: plot_loss
Uses Matplotlib to graph the train vs test Loss and saves

Parameters:
None

Returns:
None

"""
# Loss graph
def plot_loss():
    plt.plot(train_losses,'-o')
    plt.plot(eval_losses,'-o')
    plt.title('Train vs Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training','Test'])
    plt.savefig("loss.png")
    plt.close()


"""
Function: plot_accuracy()
Uses Matplotlib to graph the train vs test Accuracy and saves

Parameters: 
None

Returns:
None

"""
# Accuracy graph
def plot_accuracy():
    plt.plot(train_accu,'-o')
    plt.plot(eval_accu,'-o')
    plt.title('Train vs Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train','Valid'])
    plt.savefig("accuracy.png")
    plt.close()


"""
Function: main()
Runs all the functions

Parameters: None

Returns: 
None

"""
def main():
    
    # Check for training on GPU
    device = gpu_check()

    # Load data
    trainloader, testloader = load_data()

    # Run model
    model = cnn(device)

    # Optimize
    lossFunction, optimizer = loss_opt(model)
    
    # Train model
    train_model(model, trainloader, device, lossFunction, optimizer)

    # Loss plot
    plot_loss()

    # Accuracy plot
    plot_accuracy()

# Run all functions
main()