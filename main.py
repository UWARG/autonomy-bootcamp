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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Your working code here

# defining transform to normalize dataset from PILImage of range [0,1] to [-1,1] 
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 

BATCH_SIZE = 4

#downloading CIFAR10 dataset and transforming them
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

#creating a tuple for the image classifications
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if __name__ == '__main__':  
    # initializing np arrays to store training and validation losses and setting number of epochs
    NUM_EPOCHS = 12
    x = np.arange(1, NUM_EPOCHS+1)
    trainLossArray = np.array([])
    valLossArray = np.array([])

    # functions to show an image
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    '''
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    '''

    #creating Class for the Neural Network
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = Net()

    #defining the loss function
    criterion = nn.CrossEntropyLoss()

    #defining the optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times

        runningLoss = 0.0
        epochLoss = 0.0
        numBatches = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            numBatches += 1
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()
            # print statistics
            runningLoss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {runningLoss / 2000:.3f}')
                runningLoss = 0.0
        print("Training Epoch Loss: " + str(epochLoss/numBatches))
        trainLossArray = np.append(trainLossArray, epochLoss/numBatches) #adding training epoch loss to array

        dataiter = iter(testloader)
        images, labels = dataiter.next()
        
        '''
        # print images
        imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
        '''
        outputs = net(images)

        '''
        #output predicted class
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))
        '''
        #calculating the accuracy of the model on the dataset and logging the validation epoch losses
        correct = 0
        total = 0 
        numBatches = 0
        epochLoss = 0.0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                
                # calculate outputs by running images through the network
                outputs = net(images)

                #count the number of batches and the epoch loss
                numBatches += 1
                vloss = criterion(outputs, labels)
                epochLoss += vloss.item()

                

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("Validation Epoch Loss: " + str(epochLoss/numBatches))
        valLossArray = np.append(valLossArray, epochLoss/numBatches) #adding validation epoch loss to array
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    
    
    
    print('Finished Training')
    
 
    # plotting epoch losses
    plt.title("Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(x, trainLossArray,  label = "Training Losses", color ="green")
    plt.plot(x, valLossArray, label = "Validation Losses", color ="blue")
    plt.legend(loc="upper right")
    plt.show()
    """
    #saving and loading the trained model
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    net = Net()
    net.load_state_dict(torch.load(PATH))
    """

