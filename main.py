import torch
from matplotlib import pyplot as plt
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from matplotlib import pyplot as plt

transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])

#downloading data                                               
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#defining the model architecture
model = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1),
                      nn.ReLU(),
                      nn.MaxPool2d(2,2),
                      nn.Conv2d(32, 64, 3, padding=1),
                      nn.ReLU(),
                      nn.MaxPool2d(2,2),
                      nn.Conv2d(64, 128, 3, padding=1),
                      nn.ReLU(),
                      nn.MaxPool2d(2,2),
                      nn.Flatten(1, 3),
                      nn.Linear(128 * 4 * 4, 1000),
                      nn.ReLU(),
                      nn.Dropout(0.3),
                      nn.Linear(1000, 500),
                      nn.ReLU(),
                      nn.Dropout(0.3),
                      nn.Linear(500, 250),
                      nn.ReLU(),
                      nn.Dropout(0.3),
                      nn.Linear(250, 10),
                      nn.LogSoftmax(dim=1))

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.05)
model.to(device)

epochs = 7

#training
train_history = []
valid_history = []
for e in range(epochs):
    train_loss = 0
    valid_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model.forward(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    else:
        accuracy = 0
        model.eval()
        #validation
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                logits = model.forward(images)
                ps = torch.exp(logits)
                top_probability, top_class = ps.topk(1, dim=1)
                equal = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equal.type(torch.FloatTensor))
                loss = criterion(logits, labels)
                valid_loss += loss.item()
        model.train()
        train_history.append(train_loss/len(trainloader))
        valid_history.append(valid_loss/len(testloader))
        print(f"Epoch {e+1}/{epochs}")
        print(f"Test Accuracy {100 * accuracy/(len(testloader))}%")
        print(f"Train Loss {train_loss/len(trainloader)}")
        print(f"Valid Loss {valid_loss/len(testloader)}")

#plotting
plt.plot(train_history)
plt.plot(valid_history)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
