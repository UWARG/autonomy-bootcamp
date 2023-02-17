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

import matplotlib.pyplot as plt
import numpy as np 
import torch 
import torchvision 
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as transforms 


# Your working code here



#Composes several transforms together along with appling 2 types of transformation  :-  
#1) converting to tensor format
#2) Normalizing the dataset to get an evenly distributed set   

TRANSFORM = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

#here we are setting the batch size to only process 64 images at a time
BATCH_SIZE = 4

#The following statement is used to call the CIFAR10 dataset and apply the transformations stated in TRANSFORM
TRAINSET = torchvision.datasets.CIFAR10("./data", train = True, download  = True, transform=TRANSFORM)
# The following statement is used to load the data with the data stored in TRAINSET and in batches of 64
TRAIN_LOADER = torch.utils.data.DataLoader(TRAINSET, batch_size = BATCH_SIZE, shuffle = True)


#We do the same thing as train dataset but set shuffle and train to false since this is only used to 
#only test our CNN
TESTSET = torchvision.datasets.CIFAR10("./data", train = False, download  = True, transform=TRANSFORM)
TEST_LOADER = torch.utils.data.DataLoader(TESTSET, batch_size = BATCH_SIZE, shuffle = False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#Next we define our CNN

#We create our CNN as a class as it allows  us to tweak each and every small detail

class Net(nn.Module):

    #To initialize the CNN when the class instance is created
    def __init__(self):
        super(Net,self).__init__()   
        #We first initialize our convolutional layer which does a 2D convolution and outputs 
        #a featured map
        #2D convolution means multiplying a matrix to each individual pixel
        #Feature map is the final matrix when convolution is performed on each pixel
        #in_channels is 3 since our images will have 3 properties ,ie their RGB intensity
        # out_channels is 32 to produce 32 feature map
        # kernel_size is 3 to create a 3*3 matrix which moves through out the image 
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3) 

        #We add a max pool layer to down-size the image to allow us to better generalize the image
        self.pool = nn.MaxPool2d(2, 2) 

        #The second convolutional layer will have the same inputs as the first convolution since the 
        #the output from the first is the input for the 2nd layer
        self.conv2  = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)

        #similarly we have two more convolutional layers to increase the efficiency
        self.conv3  = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.conv4  = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3)

        #Next we create the fully connected layer in which the neuron are connected to all the 
        #neurons in the previous layer

        #The Linear function performs an operation that selects all the good features from the feature
        # map of convolutional layer 
        #We multiply 64 by 5*5 since the output of conv2 is 64 and our kernel matrix size is 5*5 
        self.fc1 = nn.Linear(64 * 5 *5, 128) 
        #The final output is 10 since we can classify an image into 10 categories
        self.fc2 = nn.Linear(128, 10) 
          

        #Next we define the forward function which determines how the image is going to flow
        #in our CNN
         
        #In our forward function  we use ReLU as our activation function to convert our sum of inputs 
        #into a binary output of 1 and 0
    def forward(self, x):
        #The image is first put in a convolutional layer and then pooled to better generalize the output
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.pool(x)

        x = self.conv3(x)
        x = self.conv4(x)

        x = self.pool(x)


        
        #The tensor which is outputted is made to change its shape using the view function
        #The -1 is used to indicate the tensor can be of any size and then get the new size of the tensor 
        #to reshape the tensor which will match the input of the FC layer
        x = x.reshape(x.size(0), -1)
 

        #Next the image is put into the FC layer and then into a ReLU function to get a final
        #binary output 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
NET = Net()
print(NET)
#Next we have to define a Loss function and an optimizor
#CrossEntropyLoss is used to train classification problems and helps in clasiffication by performing
#numerous mathematical operations. 
CRITERION = nn.CrossEntropyLoss()

#Optimizers are used to change the weights of each node to increase the accuracy
#SGD implements the stochastic gradient descent, while NET.parameters gets the changeable parameter from NET
#lr is rate at which it implements changes while momentum is used to accelerate in the right direction 
OPTIMIZER = optim.SGD(NET.parameters(),lr=0.001, weight_decay = 0.005, momentum = 0.9)

#Training and validating the neural network 

#An epoch is used to signify the number of times the neural network is going to be trained
TRAINING_LOSS = []
VALIDATION_LOSS = []
EPOCH = []
for epoch in range(20):
    #to allow easy track of the overall loss of the neural network
    TRAINING_LOSS_PER_EPOCH = []
    VALIDATION_LOSS_PER_EPOCH = []
    EPOCH.append(epoch+1)
    
    #Enumeration is used to prevent the modifications of data itself
    #In the loop below we are enumerating over the tarin_loader throug the data set where 
    #i and data is batch number and data is a list of inputs and labels  
    for i,data in enumerate(TRAIN_LOADER,0):
        #Here we are splitting the data into inputs and labels
        INPUTS,LABELS = data

        #Before we begin training will have to zero out the gradients to prevent accumulation of
        #gradients over epochs of training and the function zero_grad allows us to that
        OPTIMIZER.zero_grad()

        #Next we have to actually pass the inputs of the data set into our model
        OUTPUTS = NET(INPUTS)

        #Next we pass our outputs into a cross entropy function that allows use to predict the loss
        #of our network
        TRA_LOSS = CRITERION(OUTPUTS,LABELS)

        #adding the training loss for this batch to a list to calculate the overall loss over each epoch
        TRAINING_LOSS_PER_EPOCH.append(TRA_LOSS.item())
         
        #backward functions allows the network to perform backpropagation and adjust the gradients
        #based on the loss
        TRA_LOSS.backward()

        #after using the backward function to compute the gradients of the network we implement 
        #those changes using the optimizer we defined 
        OPTIMIZER.step() 

    #Validation of the neural network

    #Setting the neural network to validation mode
    NET.eval()

    TOTAL, CORRECT = 0, 0
    #The steps are identical with the ones for testing data but the weights are not adjusted
    with torch.no_grad():
        for i,data in enumerate(TEST_LOADER,0):
            INPUTS,LABELS = data
            OUTPUTS = NET(INPUTS)
            VAL_LOSS = CRITERION(OUTPUTS,LABELS)
            VALIDATION_LOSS_PER_EPOCH.append(VAL_LOSS.item())
            # get the predictions
            __, predicted = torch.max(OUTPUTS.data, 1)
            # update results
            TOTAL += LABELS.size(0)
            CORRECT += (predicted == LABELS).sum().item()

    TRAINING_LOSS_PER_EPOCH = np.array(TRAINING_LOSS_PER_EPOCH).mean()
    VALIDATION_LOSS_PER_EPOCH = np.array(VALIDATION_LOSS_PER_EPOCH).mean()
    

    
    TRAINING_LOSS.append(TRAINING_LOSS_PER_EPOCH)
    VALIDATION_LOSS.append(VALIDATION_LOSS_PER_EPOCH)


    print(epoch+1,"Training loss:- ",TRAINING_LOSS_PER_EPOCH,"Validation loss:- ", VALIDATION_LOSS_PER_EPOCH ,"Accuracy of the model:-",100 * CORRECT // TOTAL,"%")


#Plotting the losses over epochs
plt.plot(EPOCH,TRAINING_LOSS,label = "Training Loss") 
plt.plot(EPOCH,VALIDATION_LOSS,label = "Validation Loss")   
plt.xlabel(" Epochs ")
plt.ylabel(" Losses ")   
plt.legend()
plt.show()

#Testing the Neural Network
dataiter = iter(TEST_LOADER)
images, labels = next(dataiter)

print('GroundTruth: ', ' '.join('%s' % classes[labels[j]] for j in range(4)))

outputs = NET(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%s' % classes[predicted[j]]
                              for j in range(4)))
