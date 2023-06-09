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

# Basic libraries
import numpy as np
import matplotlib.pyplot as plt

# PyTorch libraries
import torch
import torchvision
import torchvision.transforms as transforms

#Neural network libraries
import torch.nn as nn
import torch.nn.functional as F

# #Creating optimizer
import torch.optim as optim

# For terminating program (useful for stopping before training is complete and seeing entire result)
import signal
import sys

# Your working code here


def imshow(img):
    """
    Shows an image using matplotlib

    Args
    -------------
        img (torch.Tensor): The input image
    
    Returns
    ------------
        None
    """
    img=img.detach()
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
    

##--------------------------Neural Network Code ----------------------------##

# Creating a CNN

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Creates a Convolutional neural network for CIFAR-10 dataset with the following structure:
    - First convolutional layer (conv1)
    - Second convolutional layer (conv2)
    - Third convolutional layer (conv3)
    - Three fully connected layers (fc1, fc2, fc3)
    
    Attributes:
        conv1 : nn.Conv2d
        conv2 : nn.Conv2d
        conv3 : nn.Conv2d
        fc1   : nn.Linear
        fc2   : nn.Linear
        fc3   : nn.Linear
    
    Methods:
        __init__()
            Creates the structure of the neural network   
        forward(x) 
            Forward pass of the neural network    
    """
    def __init__(self):
        super(Net, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # Applies convolution by sliding over the input image over 3 channels (RGB)
        # and produces 32 output channels
        # Window size is 3x3
        # Padding is added to maintain input size
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # Applies convolution by sliding over the input image over 32 channels (output of conv1)
        # and produces 64 output channels
        # Window size is 3x3
        # Padding is added to maintain input size
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # Applies convolution by sliding over the input image over 64 channels (output of conv2)
        # and produces 128 output channels
        # Window size is 3x3
        # Padding is added to maintain input size
        
        # First fully connected layer
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        # Input size = 128 * 4 * 4 (images from conv3)
        # Output size = 512
        
        # Second fully connected layer
        self.fc2 = nn.Linear(512, 256)
        # Input size = 512 (from fc1)
        # Output size = 256
        
        # Third fully connected layer
        self.fc3 = nn.Linear(256, 10)
        # Input size = 256 (from fc2)
        # Output size = 10 (deciding output classes)

    def forward(self, x):
        """
        Forward pass of the neural network

        Args:
            x (torch.Tensor): Input image

        Returns:
            x (torch.Tensor): Output image
        """
        # Applies conv1 to x, followed by ReLU and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # Applies conv2 to x, followed by ReLU and max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # Applies conv3 to x, followed by ReLU and max pooling
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        # Reshapes the tensor x
        x = x.view(x.size(0), -1)
        
        # Applies fc1, fc2, fc3
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


#Training the neural network:

def train(net,trainloader,optimizer,criterion):
    """
    Trains the neural network
    It loops keeps repeating the training until the user terminates the program

    Args:
        net (Net): The convolutional neural network
        trainloader (DataLoader): Stores the training data in iterable form
        optimizer (torch.optim): Optimizer used for training 
        criterion (torch.loss): The loss function 
    
    Returns:
        None
    """
    global losses_training,accuracy_training,losses_testing,accuracy_testing #used for the end
    losses_training=[]
    accuracy_training=[]
    
    losses_testing=[]
    accuracy_testing=[]
        
    print("Training:")
    epoch=1
    while True: #loop over dataset until terminated
        #this is for printing statistics
        running_loss=0.0
        correct = 0
        for i,data in enumerate(trainloader,0):
            #Getting inputs
            inputs,labels = data
            
            #Making the parameters zero
            optimizer.zero_grad()
            
            #Forward pass
            outputs = net(inputs)
            
            #Calculating loss
            loss = criterion(outputs,labels)
            
            #Backpropogation
            loss.backward()
            
            #Updating parameters
            optimizer.step()
            
            #Getting statistics
            running_loss += loss.item()
            _,predicted = torch.max(outputs.data,1)
            correct += (predicted==labels).sum().item()
        #print(i)
        
        # Statistics for each epoch
        print(f'Epoch: {epoch}, Loss: {running_loss/(50000/BATCH_SIZE):.6f}, Accuracy: {100*correct/(50000):.6f}%')
        losses_training.append(running_loss/12000)
        accuracy_training.append(100*correct/50000)
        correct=0
        running_loss=0.0
        
        # Statistics for testing data
        cor,los=test(net,testloader,criterion)  
        losses_testing.append(los/(10000/BATCH_SIZE))
        accuracy_testing.append(100*cor/10000)
        
        epoch+=1

    
# Sample tests:

def sample_test(net):
    """
    Tests on 4 random images from test data

    Args:
        net (Net): The trained CNN
    """
    print("\nTest on 4 random images")
    do = input("Do you want to test and see sample images?(y/n):")

    if do=='y':
        dataiter = iter(testloader)
        images,labels  = next(dataiter)

        print("Actual:", ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
        outputs = net(images)
        _, predicted = torch.max(outputs, dim=1)
        print("Predicted:", ' '.join(f'{classes[predicted[j]]}' for j in range(4)))
        imshow(torchvision.utils.make_grid(images))
            
        
# Testing on overall dataset test:

def test(net,testloader,criterion):
    """
    Tests the neural network

    Args:
        net (Net): The trained conolutional neural network
        epochs (int): Number of times we iterate through testing data
        testloader (DataLoader): Stores the testing data in iterable form
    
    Returns:
        correct (int): Number of correct predictions
        running_loss(float): Loss of the testing data
    """
    correct=0
    running_loss=0.0
    with torch.no_grad():
        for data in testloader:
            images,labels=data
            outputs = net(images)

            _,predicted = torch.max(outputs.data,1)
            correct+=(predicted==labels).sum().item()
            
            loss=criterion(outputs,labels)
            running_loss+=loss.item()

    print(f'Testing... Loss: {running_loss/(10000/BATCH_SIZE):.6f}, Accuracy: {100*correct/10000:.6f}%')
    return correct,running_loss


##-------------------------------------Plotting-------------------------------------##

def accuracy_plots(train_accuracy,test_accuracy):
    """
    Plots the accuracy of the training and testing data

    Args:
        train_accuracy (list): List of accuracies for each epoch of training
        test_accuracy (list): List of accuracies for each epoch of testing
    """
    plt.plot(train_accuracy,label="Training")
    plt.plot(test_accuracy,label="Testing")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of training and testing data")
    plt.legend()
    plt.show()

def losses_plots(train_loss, test_loss):
    """
    Plots the loss of the training and testing data

    Args:
        train_loss (list): List of losses for each epoch of training
        test_loss (list): List of losses for each epoch of testing
    """
    plt.plot(train_loss,label="Training")
    plt.plot(test_loss,label="Testing")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss of training and testing data")
    plt.legend()
    plt.show()

##-------------------------------------Main Program-------------------------------------##
def on_termination(signal,frame):
    """This function is called when the program is to be terminated"""
    accuracy_plots(accuracy_training,accuracy_testing)
    losses_plots(losses_training,losses_testing)
    if (input("End?(y/n):")=="y"):
        print("Finished training")
        sys.exit(0)

signal.signal(signal.SIGINT, on_termination)   # Handles Ctrl+C
signal.signal(signal.SIGTERM, on_termination)  # Handles termination signal


## Main program

if __name__=="__main__":

    # Extracting data from CIFAR-10

    # Convert output of images from torchvision dataset from [0,1] to Tensors of range [-1,1]
    # Plus some data augmentation for the training data
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4)
    ])
    
    transform_test= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    BATCH_SIZE = 16

    # These are the images which will be used for training
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_train)

    # Provides an iterable for training
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                            num_workers=0)

    # These are the images which will be used for testing
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                        transform=transform_test)

    # Provides an iterable for testing
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True,
                                            num_workers=0)

    # Classes of images
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Creating the neural network
    net = Net()
    
    #Loss function and optimizer
    LEARNING_RATE = 0.0005
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=LEARNING_RATE)

    #Training the neural network
    #Will run indefinitely until terminated by user
    train(net,trainloader,optimizer,criterion)
