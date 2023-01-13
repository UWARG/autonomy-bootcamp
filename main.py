import torch
import numpy as np

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

#if gpu available run training on gpu else cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#establishing number of epocs and learning rate 
num_epochs = 10
batch_size = 4
learning_rate = 0.001

#normalizing [0, 1] pixels to be in range -1, 1 to prevent exploding gradeints
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 

 #downloading/loading datasets
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#identifying classes to be used 
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#convolutional neural network implementation 
class ConvNet(nn.Module): 
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            #settuing up the first layer in cnn
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1),
            #batch normalization to make training more efficient
            nn.BatchNorm2d(64),
            #ReLU activation function (all negative values in tensors changed to 0)
            nn.ReLU(inplace = True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #maxpooling used to reduce feature map dimensions + computation
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4), 
            #reshaape tensor to 1dimension
            nn.Flatten(), 
            #last layer reduces 512 channels in to 10 outputs to match output classes
            nn.Linear(512, 10) 
        )

    def forward(self, x):
        #forward propagation
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x

#storing loss in training datasets and validation datasets as y axis
y_loss = {} 
y_loss['train'] = []
y_loss['val'] = []

#storing epoch number as x axis
x_epoch = [] 
fig = plt.figure() #plotting train and val loss
ax0 = fig.add_subplot(121, title = "loss")

#draw function for curve
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()

model = ConvNet().to(device)

#using Cross Entropy Loss as loss metric
criterion = nn.CrossEntropyLoss()
#stochastic gradient descend 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  


#running CNN on training data and validation data and recording values for axis 
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            # Set model to training mode when using training data
            model.train(True)  
        else:
            # Set model to evaluate mode when using validation data (otherwise could lead to overfitting)
            model.train(False)  
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #printing out loss results after specific epoch
            if (i+1) % 2000 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        #addending the loss values to y-axis of graph
        y_loss[phase].append(loss.item())
        if phase == 'val':
            #after completing validation set draw the curves
            draw_curve(epoch)


print('Finished Training')

with torch.no_grad(): #calculating accuracy by testing against validation set
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0*n_correct/n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0*n_class_correct[i]/n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

fig.savefig("C:/Users/thoma/OneDrive/Documents/GitHub/computer-vision-bootcamp/plot.png")