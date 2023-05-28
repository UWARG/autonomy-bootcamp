"""Creates an appropriate model for the CIFAR-10 dataset, using CNN. Utilizes data augmentation and dropouts"""
# Import whatever libraries/modules you need

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Your working code here

## Class and Function Definitions.
class ConvNet(nn.Module):
    """Contains two functions:
    i) Creates the Convolutional Network with appropriate Conv2d and MaxPooling layers
    ii) Conducts the forward propogation each propogation"""

    def __init__(self):
        """Creates a Convolutional Network with 2 colvolutional and 1 maxpooling layer, followed by
        2 linear layers with Dropouts"""

        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.maxpool = nn.MaxPool2d(2, 2)
        self.drpt = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 6 * 6, 500)
        self.fc2 = nn.Linear(500, 10) 


    def forward(self, x):
        """Forward Propogation, uses 2 convolutional layers that use maxpooling. Then uses 2 linear
        layer, the first one of which uses dropouts with p = 0.5."""
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.drpt(self.fc1(x)))
        x = self.fc2(x)## No Softmax required, already included 
        ## in cross entropy loss_function
        return x


def train(model, data_loader, loss_function, optimizer):
    """Trains the model using crossentropy as loss function, prints the accuracy based on proportion
    of correct predictions."""
    n_total_steps = len(data_loader.dataset)
    size_dataset = len(train_loader)
    model.train()
    epoch_loss = 0
    correct_pred = 0

    for i, (images, labels) in enumerate(train_loader):
        ## To get GPU support for the images and labels. Assigns the Tensors to GPU
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        ## Forward pass
        outputs = model(images)
        loss = loss_function(outputs, labels) ## Finding the amount of loss. The more the loss, the more the calibration

        ## Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        correct_pred += (outputs.argmax(1) == labels).type(torch.float).sum().item()

    epoch_loss /= size_dataset
    correct_pred /= n_total_steps
    acc = correct_pred * 100

    print(f'Training Info: Accuracy: {acc:3f}%, Loss: {epoch_loss:4f}')

    train_loss.append(epoch_loss)
    train_acc.append(correct_pred)


def test(model, data_loader, loss_function):
    """Tests the validation set and gives the accuracy and loss score. """

    size_batch = len(test_loader)
    size_dataset = len(test_dataset)
    model.eval()
    loss = 0
    correct_pred = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            correct_pred += (outputs.argmax(1) == labels).type(torch.float).sum().item()
            epoch_loss = loss_function(outputs, labels).item()
            loss += epoch_loss

        loss /= size_batch
        correct_pred /= size_dataset
        acc = 100.0 * correct_pred

        print(f'Accuracy of the network: {acc:4f}%, Loss: {loss:4f}')

        test_acc.append(correct_pred)
        test_loss.append(loss)


def accuracy_plot():
    """Plots the accuracy graph for the training and validation sets using matplotlib"""

    plt.plot(train_acc, '-o')
    plt.plot(test_acc, '-o')
    plt.title('Training vs Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (Probability)')
    plt.legend(['Training', 'Validation'])
    plt.show()


def loss_plot():
    """Plots the loss graph for the training and validation sets using matplotlib"""

    plt.plot(train_loss, '-o')
    plt.plot(test_loss, '-o')
    plt.title('Training vs Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross Entropy)')
    plt.legend(['Training', 'Validation'])
    plt.show()


## DEVICE configuration. The code works much faster when running on the GPU.
## torch.cuda.is_available() gives True only when proper gpu support libraries for 
## cuda have been installed.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Main parameters 
NUM_EPOCHS = 100 ## These are the number of epochs. Make it whatever you want. Default value is 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001 ## Default value 0.001. Smaller steps ensure no overshooting, however, can decrease 
## loss function very slowly.

## Images of the dataset are Tensors with each value being from 0 to 1. We are transforming them to Tensors
## having value from -1 to 1. I have also done data augmentation, to create a better model that can have a 
## better accuracy on the validation set. However, that requires much more computational power.
transform = transforms.Compose([#transforms.ToPILImage(),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ColorJitter(brightness=0.5),
                                transforms.RandomRotation(degrees=45),
                                transforms.RandomVerticalFlip(p=0.05),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

## Dataset stores the samples and their corresponding labels
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)


## DataLoader wraps an iterable around the Dataset to enable easy access to the samples. It simplifies
## the process for python.
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=BATCH_SIZE,
                                            num_workers=2,
                                            shuffle=False) ## We don't wish to shuffle this. The images should
## have their designated labels only

model = ConvNet().to(DEVICE) ## To get GPU support

## Keeping track of training accuracy and loss through the model
train_loss = []
train_acc = []

test_loss = []
test_acc = []

loss_function = nn.CrossEntropyLoss() ## This is multiclass problem. Cross Entropy is the best fit.
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    print(f'Epoch: {epoch}/{NUM_EPOCHS}')
    print('<-------------->')
    train(model, train_loader, loss_function, optimizer)
    test(model, test_loader, loss_function)
    print('\n')

accuracy_plot()
loss_plot()
