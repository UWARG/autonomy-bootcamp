# Import whatever libraries/modules you need
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Your working code here

#Defining a Convolutional Neural Network to classify images
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)


        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu((self.conv1(x)))
        x = self.bn2(self.pool(F.relu(self.conv2(x)))) #64 x 16 x 16
        x = F.relu((self.conv3(x)))
        x = self.bn4(self.pool(F.relu(self.conv4(x)))) #128 x 8 x 8
        x = F.relu((self.conv5(x)))
        x = self.bn6(self.pool(F.relu(self.conv6(x)))) #256 x 4 x 4
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Calling the function loads and normalizes the CIFAR10 dataset with torchvision
def load():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 10

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

def get_loss():
    return nn.CrossEntropyLoss()

def get_optimizer(net):
    return optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def plot(value, epoch):
    plt.plot(value)
    plt.xlabel("EPOCH")
    plt.xticks(np.arange(0,epoch ,1))
    plt.title("Validation Accuracy")
    plt.show()


def train(net, train_loader, test_loader, epochs):
    criterion = get_loss()
    optimizer = get_optimizer(net)
    accuracies = np.zeros(epochs)

    for epoch in range(epochs): 
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            #optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        #getting and printing the epoch, loss and accuracy
        running_loss += loss.item()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracies[epoch] = 100 * correct // total
            print(f'Epoch:{epoch + 1} Loss: {running_loss:.3f} Accuracy: {accuracies[epoch]}')
            running_loss = 0.0
    print('Finished Training')
    return accuracies

if __name__ == "__main__":
    __epoch = 5
    __train_loader, __test_loader = load()
    net = Net()
    accuracies = train(net, __train_loader, __test_loader, __epoch)
    plot(accuracies, __epoch)
