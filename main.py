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
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import Net

# Your working code here

torch.manual_seed(0) # to reproduce the situation every time

# keep track of the training accuracy and losses
train_accu = []
train_losses = []

# keep track of the testing accuracy and losses
eval_accu = []
eval_losses = []

# method to extract the data
def extract_data():
    """
        Function: extract_data()

        Extract the CIFAR-10 dataset from torchvision.datasets

        Parameters:
        None

        Returns:
        trainloader: DataLoader - An iterable for the training dataset with the specified batch size
        testloader: DataLoader - An iterable for the testing dataset with the specified batch size 
    """
    # creating the transformation of the data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ]
    )

    batch_size = 128 # batch size

    # extracting the trainset and creating the trainloader
    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    # extracting the testset and creating the testloader
    testset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    return trainloader, testloader

def optimizer_loss_function_generator(net: Net):
    """
        Function: optimizer_loss_function_generator(net)

        Creates and returns the loss function and the optimizer for the model

        Parameters:
            @param net: Net - The Vonvolutional Neural Network (CNN) model developed. 
        
        Return:
            criterion: CrossEntropyLoss - the Cross Entropy Loss function
            optimizer: SGD - the Stochiostic Gradient Descent optimization function
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    return criterion, optimizer

def train(net: Net, criterion: nn.CrossEntropyLoss, optimizer: optim.SGD, trainloader: DataLoader):
    """
        Function: train(net, criterion, optimizer, trainloader)

        Function to train the model based on the given parameters for 1 epoch

        Parameters:
            @param net: Net - The Vonvolutional Neural Network (CNN) model developed. 
            @param criterion: CrossEntropyLoss - the Cross Entropy Loss function 
            @param optimizer: SGD - the Stochiostic Gradient Descent optimization function 
            @param trainloader: DataLoader - An iterable for the training dataset with the specified batch size 
        
        Side Effects: 
            Displays the train loss and the train accuracy for the particular epoch. 
        
        Return:
            None
    """
    total = 0
    correct = 0
    running_loss = 0.0

    for data in tqdm(trainloader):
        # separating the input and label data
        inputs, labels = data

        optimizer.zero_grad() # zeroes the current gradient

        outputs = net(inputs) # finds the output based on the current weights and biases

        # performing gradient descent on the loss function to improve the weights and biases
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # keeping track of value of loss function per batch
        running_loss += loss.item()

        # keeping track of the training accuracy per batch
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    # train loss
    train_loss = running_loss / len(trainloader)
    train_accuracy = (100 * correct) / total

    # Adding to the list for plotting
    train_accu.append(train_accuracy)
    train_losses.append(train_loss)

    # Printing the progress to decimal places
    print(f'Train Accuracy: {train_accuracy:.2f}| Train Loss: {train_loss:.2f}')
        
def test(net: Net,  criterion: nn.CrossEntropyLoss, testloader: DataLoader):
    """
        Function: test(net, criterion, trainloader)

        Function to test the model based on the given parameters for 1 epoch

        Parameters:
            @param net: Net - The Vonvolutional Neural Network (CNN) model developed. 
            @param criterion: CrossEntropyLoss - the Cross Entropy Loss function 
            @param testloader: DataLoader - An iterable for the testing dataset with the specified batch size 
        
        Side Effects: 
            Displays the test loss and the test accuracy for the particular epoch. 
        
        Return:
            None
    """
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad(): # stopping gradient descent
        for data in tqdm(testloader):
            # separating the input and label data
            images, labels = data

            outputs = net(images) # finds the output based on the current weights and biases

            # calculating and adding up the value of the loss function per batch
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # keeping track of the testing accuracy per batch 
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # train loss
    test_loss = running_loss / len(testloader)
    test_accuracy = (100 * correct) / total

    # Adding to the list for plotting
    eval_accu.append(test_accuracy)
    eval_losses.append(test_loss)

    # Printing the progress to decimal places
    print(f'Test Accuracy: {test_accuracy:.2f}| Test Loss: {test_loss:.2f}')

def plot_loss():
    """
        Function: plot_loss()

        Function to plot the line chart for loss vs epoch of training and testing datasets.

        Parameters:
           None
        
        Side Effects: 
            Saves the plot as a png image in the respository.
        
        Return:
            None
    """
    plt.plot(train_losses,'-o')
    plt.plot(eval_losses,'-o')
    plt.title('Train vs Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training','Test'])
    plt.savefig("loss_comparison.png")
    plt.close()

def plot_accuracy():
    """
        Function: plot_accuracy()

        Function to plot the line chart for accuracy vs epoch of training and testing datasets.

        Parameters:
           None
        
        Side Effects: 
            Saves the plot as a png image in the respository.
        
        Return:
            None
    """
    plt.plot(train_accu,'-o')
    plt.plot(eval_accu,'-o')
    plt.title('Train vs Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train','Valid'])
    plt.savefig("accuracy_comparison.png")
    plt.close()

def main():
    """
        Function: main()

        Runs all of the other functions. 

        Parameters:
            None

        Side Effects:
            prints the labels/classes for the output. 
        
        Return:
            None
    """
    # creating the object of the model class Net
    net = Net()

    # extracting training and testing data
    trainloader, testloader = extract_data()

    # defining the final output classes of the data (for reference only)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(f"Classes are as follows: {classes}")

    # creating the optimizer and loss function for the model to use
    criterion, optimizer = optimizer_loss_function_generator(net=net)

    epochs = 10 # number of epochs

    # Loop through all the epochs to train and test the model
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}")
        train(net=net, criterion=criterion, optimizer=optimizer, trainloader=trainloader)
        test(net=net, criterion=criterion, testloader=testloader)
    
    # plot and save the loss vs epoch graph for training and testing
    plot_loss()
    
    # plot and save the accuracy vs epoch graph for training and testing
    plot_accuracy()
    

if __name__ == '__main__':
    main()
