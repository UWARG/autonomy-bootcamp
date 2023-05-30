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
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

#Neural network libraries
import torch.nn as nn
import torch.nn.functional as F

# #Creating optimizer
import torch.optim as optim

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

# STEP 1: Extracting data from CIFAR-10

# Convert output of images from torchvision dataset from [0,1] to Tensors of range [-1,1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

BATCH_SIZE = 4

# These are the images which will be used for training
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                        transform=transform)

# Provides an iterable for training
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=0)
# num_workers is set to 0 since we want data loading
# to be performed as the main process
# without that it gives an error of multiprocessing

# These are the images which will be used for testing
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                       transform=transform)

# Provides an iterable for testing
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True,
                                         num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# STEP 2: Creating a CNN

class Net(nn.Module):
    """
    Creates a Convolutional neural network with the following structure:
    - First convolutional layer (conv1)
    - Pooling layer (pool)
    - Second convolutional layer (conv2)
    - Three fully connected layers (fc1, fc2, fc3)
    
    Attributes:
        conv1 : nn.Conv2d
        pool  : nn.MaxPool2d
        conv2 : nn.Conv2d
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
        super().__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Applies convolution by sliding over the input image over 3 channels (RGB)
        # and produces 6 output channels
        # Window size is 5x5

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Maxpooling takes the maximum values from each local region and reduces its spatial dimensions

        # Second convolutional layer
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Applies convolution by sliding over the input image over 6 channels (output of conv1)
        # and produces 16 output channels
        # Window size is 5x5
        
        self.drpt = nn.Dropout(p=0.5)

        # First fully connected layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # Input size = 16 * 5 * 5 (images from conv2)
        # Output size = 120

        # Second fully connected layer
        self.fc2 = nn.Linear(120, 84)
        # Input size = 120 (from fc1)
        # Output size = 84

        # Third fully connected layer
        self.fc3 = nn.Linear(84, 10)
        # Input size = 84 (from fc2)
        # Output size = 10 (deciding output classes)

    def forward(self, x):
        """
        Forward pass of the neural network

        Args:
            x (torch.Tensor): Input image

        Returns:
            x (torch.Tensor): Output image
        """
        # Applies conv1 to x, followed by ReLU, followed by maxpooling
        x = self.pool(F.relu(self.conv1(x)))

        # Applies conv2 to x, followed by ReLU, followed by maxpooling
        x = self.pool(F.relu(self.conv2(x)))

        # Reshapes the tensor x
        x = x.view(x.size(0), -1)

        # Applies fc1, fc2, fc3
        x = F.relu(self.drpt(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#STEP 3: Training the network

def train(net,epochs,trainloader,optimizer,criterion):
    """Trains the neural network

    Args:
        net (Net): The convolutional neural network
        epochs (int): Number of times we iterate through training data
        trainloader (DataLoader): Stores the training data in iterable form
        optimizer (torch.optim): Optimizer used for training (SGD in this case)
        criterion (torch.loss): The loss function (CrossEntropyLoss in this case)
    
    Returns:
        net (Net): The trained convolutional neural network
        losses_training(list): List of losses for each epoch
        accuracy_training(list): List of accuracies for each epoch
    """
    losses_training=[]
    accuracy_training=[]
    
    losses_testing=[]
    accuracy_testing=[]

    print("Training:")
    for epoch in range(epochs): #loop over dataset multiple times
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
            
            #Printing statistics
            running_loss += loss.item()
            _,predicted = torch.max(outputs.data,1)
            correct += (predicted==labels).sum().item()
        #print(i)
        
        #Statistics for each epoch
        print(f'Epoch: {epoch+1}, Loss: {running_loss/12000:.6f}, Accuracy: {100*correct/(50000):.6f}%')
        losses_training.append(running_loss/12000)
        accuracy_training.append(100*correct/50000)
        correct=0
        running_loss=0.0
        
        cor,los=test(net,testloader,criterion)  
        losses_testing.append(los/(10000/BATCH_SIZE))
        accuracy_testing.append(100*cor/10000)
        
        #Save trained model
        PATH = './cifar_net.pth'
        torch.save(net.state_dict(),PATH)
                

    print("Finished training")
    return net,losses_training,accuracy_training,losses_testing,accuracy_testing

    

#STEP 4 : Testing the network

#Sample tests:
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
            
        
#Overall dataset test:


def test(net,testloader,criterion):
    """
    Tests the neural network

    Args:
        net (Net): The trained conolutional neural network
        epochs (int): Number of times we iterate through testing data
        testloader (DataLoader): Stores the testing data in iterable form
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

    print(f'Testing: Loss: {running_loss/(10000/BATCH_SIZE):.6f}, Accuracy: {100*correct/10000:.6f}%')
    return correct,running_loss


#Plotting

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


## Main program with interface

def main():
    """
    Main function
    """
    net = Net()
    
    #Loss function and optimizer
    LEARNING_RATE = 0.001
    criterion = nn.CrossEntropyLoss() #good loss function for this model
    optimizer = optim.SGD(net.parameters(),lr=LEARNING_RATE,momentum=0.9)
    
    EPOCHS=int(input("How many epochs: "))
    
    net,losses_training,accuracy_training, losses_testing,accuracy_testing = train(net,EPOCHS,trainloader,optimizer,criterion)
    accuracy_plots(accuracy_training,accuracy_testing)
    losses_plots(losses_training,losses_testing)

main()