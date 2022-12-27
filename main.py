# Import whatever libraries/modules you need
import torch
from torch import optim, nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

# Your working code here

# Hyperparameters
BATCH_SIZE = 16
EPOCH_NUM = 10 # computer go boom boom
MAX_LR = 0.01 
MOMENTUM = 1.2
DECAY = 0.0001 

# use cuda if cuda available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Transform data to tensor and normalize RGB values
transform = Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Dataset
data_train = CIFAR10(root="./", download = True, train = True, transform = transform)
data_test = CIFAR10(root="./", download = True, train = False, transform = transform)

# Dataloader
train_loader = DataLoader(data_train, batch_size = BATCH_SIZE)
test_loader = DataLoader(data_test, batch_size = BATCH_SIZE)

# Model for prediction
class CIFAR10(nn.Module):
  def __init__(self):
    super().__init__()
    self.neural_net = nn.Sequential(
        # 2d CNN layers
        nn.Conv2d(3,32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(1024), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(2048), nn.ReLU(), nn.MaxPool2d(4, 4),
        nn.Flatten(),
        # Final flat layers
        nn.Linear(2048, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 10)
    )
  def forward(self, x):
    return self.neural_net(x)

# Create the model
model = CIFAR10().to(device)
model.to(device)

# Loss calculator
lossCalc = nn.CrossEntropyLoss()
# SGD Optimizer
optimizer = optim.SGD(model.parameters(), lr= MAX_LR, momentum= MOMENTUM ,weight_decay= DECAY)
# Scheduler for variable learning rate
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr= MAX_LR, steps_per_epoch= len(train_loader), epochs= EPOCH_NUM)

# Lists to store training and validation loss
training_loss = []
validation_loss = []

# Training Loop
for epoch in range(EPOCH_NUM):

    # Training loop
    running_loss = 0.0
    model.train()
    for data in train_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        # Resets gradient for each loop
        optimizer.zero_grad()
        outputs = model(inputs)
        # calculate loss
        loss = lossCalc(outputs, labels)
        # Sum the loss
        running_loss += loss.item()
        # Back propagation
        loss.backward()
        # Parameter update for optimizer and scheduler
        optimizer.step()
        scheduler.step()
    
    training_loss.append( round((running_loss/len(train_loader)),3) )

    # Validation loop
    total = 0
    correct = 0
    valid_loss = 0.0
    model.eval()
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        loss = lossCalc(outputs, labels)
        valid_loss += loss.item()
        i, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    validation_loss.append( round((valid_loss/len(test_loader)),3) )
    print(f'Accuracy: {round((100 * correct / total),2)} % ')


# Final pass for 
correct = 0
total = 0
# no gradiant calculation needed for validation
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the model
        outputs = model(images)
        i, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Output to terminal results
print(f'Final Accuracy: {round((100 * correct / total),2)} %')
print("Training loss: ",training_loss)
print("Validation loss: ",validation_loss)

epoch_legends = [i for i in range(1,EPOCH_NUM+1)]

# Plot using matplotlib
plt.plot(epoch_legends, training_loss, label = "Training Loss")
plt.plot(epoch_legends, validation_loss, label = "Validation Loss")
plt.xlabel("Epoch Number", fontsize=15)
plt.ylabel("Losses", fontsize=15)
plt.legend()
plt.show()