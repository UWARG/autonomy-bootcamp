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

import matplotlib.pyplot as plt # used for plotting 
import numpy as np # used for transfromations 

import torch # PyTorch
import torchvision # used for datasets 
import torchvision.transforms as transforms # used for image transformations
import torch.nn as nn # used for neural networks 
import torch.nn.functional as F # used for convolutional operations
import torch.optim as optim # used for optimization algorithms 

# Your working code here
if __name__ == '__main__':

    # composes several transforms together 
    transform = transforms.Compose( 
        [transforms.ToTensor(), # converts type images to tensors 
        transforms.RandomHorizontalFlip()])
    

    # batch size, number of training samples in one interation
    batch_size = 5

    # 2 workers simultaneosuly putting data into computer 
    num_workers = 2

    # num of epochs
    EPOCHS = 20

    # load train data 
    # root creates data folder, set download and train as true, and pass in previously defined transformation
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # to load data, set shuffle to true to reshuffle data at every epoch
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # load test data, same as before except train and shuffle set to false bc this is used to test
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # define classes to avoid duplicates 
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the CNN
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__() #initialize the network
            self.conv1 = nn.Conv2d(3, 32, 3) # in_channels = 3 bc RGB, out_channels = 32 feature maps, kernel_size = 3x3 sqaure convolutional kernel
            self.pool = nn.MaxPool2d(2, 2) # turn image into 2x2 to retain important features
            self.conv2 = nn.Conv2d(32, 64, 3) # output from first layer is input for second layer, goes in succesively
            self.conv3 = nn.Conv2d(64, 128, 3)
            self.fc1 = nn.Linear(128 * 2 * 2, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 10)
            self.dropout1 = nn.Dropout(p=0.2, inplace=False) # helps with overfitting

        def forward(self, x): #forward propogation algorithim
            x = self.pool(F.relu(self.conv1(x))) # passes conv1 into ReLU activation function (for non-linearity)
            x = self.dropout1(x)
            x = self.pool(F.relu(self.conv2(x)))
            x = self.dropout1(x)
            x = self.pool(F.relu(self.conv3(x)))
            x = self.dropout1(x)
            x = x.view(-1, 128 * 2 * 2) # reshape tensors, flatten output from conv layer  
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x) #output layer 
            return x
    
    net = Net()

    # leverage GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # define loss function and optimizer 
    criterion = nn.CrossEntropyLoss() # combine log softmax and negative log-likelihood
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # implements stochastic gradient descent 

    # train the network 
    losses = []
    accuracies = []
    for epoch in range(EPOCHS):  # loop over entire train set twice 
        
        # running loss
        epoch_loss = 0
        total = 0
        correct =0

        for i, data in enumerate(trainloader, 0): # loop to enumerate over batches from trainloader 
            inputs, labels = data # split data into input and label objects
            
            optimizer.zero_grad() # zero out gradients 
            outputs = net(inputs) # pass inputs into neural network
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            loss = criterion(outputs, labels) # pass outputs into loss function, labels is target
            loss.backward() # backpropogation by computing gradient based on loss
            optimizer.step() # updates paramater values 
            epoch_loss += loss.item()

        # loss per epoch
        print(f'Epoch {epoch+1} of {EPOCHS}   Accuracy: {correct* 100/ total}%  Loss: {epoch_loss/len(trainloader)}')

        # accuracy per epoch
        losses.append(epoch_loss/len(trainloader))
        accuracies.append(correct* 100/ total)   

    # plots for loss/accuracy
    plt.plot(losses)
    plt.title('Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(accuracies)
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    correct = 0
    total = 0
    with torch.no_grad(): # dont need autograd, save memory
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of network: %d %%' % (100 * correct / total))

    
    inputs, labels = data[0].to(device), data[1].to(device)
