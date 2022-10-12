import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Net Class. 
    parent: 
        torch.nn.Module
    
    Used to create the model involving Convolutional Neural Network (CNNs)

    Attributes: 
        conv1: 2d Convulated Layer 1
        conv2: 2d Convulated Layer 2
        conv3: 2d Convulated Layer 3 
        conv4: 2d Convulated Layer 4 
        conv5: 2d Convulated Layer 5 

        dense1: Linear Layer 1
        dense2: Linear Layer 2
        dense3: Linear Layer 3
    
    Constructor: 
        __init__()
    
    Methods:
        forward()
    """
    def __init__(self):
        """
            Constructor: __init__()

            Defines the layers involved in building the Neural Network of the model
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)

        # Linear (Dense)
        self.dense1 = nn.Linear(512, 120)
        self.dense2 = nn.Linear(120, 32)
        self.dense3 = nn.Linear(32, 10)
    
    def forward(self, x):
        """
            Function: forward()

            Defines the layers with their respective activation functions
                and properties as the forward propogation is done. 
            
            Parameters:
                @param x: Tensor - the input provided to the input layer for making the decision
            
            Returns:
                x: Tensor - returns the output of the model in the form of a tensor
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x
