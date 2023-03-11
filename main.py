import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader

import numpy
from matplotlib import pyplot as plt

from timeit import default_timer as timer

# Device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}\n')

# Load data
train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=ToTensor(), target_transform=None)
test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=ToTensor(), target_transform=None)
class_names = train_data.classes

# Create dataloaders
BATCH_SIZE = 8
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

# Using the Tiny VGG architecture
# 2 convolutional blocks and one classifier block
# Each convoltional block has 2 convolutional layers, with ReLU activation layers in between and a pooling layer at the end
# The classification block classifies the image
class Tiny_VGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels = input_shape, out_channels = hidden_units,
            kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units,
            kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units * 2,
            kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units * 2, out_channels = hidden_units * 2,
            kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = hidden_units*2*7*7, out_features = output_shape))
    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

# Instantiate model
model = Tiny_VGG(input_shape = 3, hidden_units = 64, output_shape = len(class_names)).to(device)

# Set loss and optimizer functions
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum = 0.9)

# Define accuracy function
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# Train model over 10 epochs, record the results, and time how long it took
start_time = timer()
epochs = 10
results = {'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc' : []}
print('\nStart Training\n')
for epoch in range(epochs):    
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(train_dataloader):
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)
        
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true = y, y_pred = y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            test_pred = model(X_test)

            test_loss += loss_fn(test_pred, y_test)
            test_acc += accuracy_fn(y_true = y_test, y_pred = test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    print(f'Epoch: {epoch}')
    print(f'Train Loss: {train_loss}, Train Acc: {train_acc} | Test Loss: {test_loss}, Test Acc: {test_acc}\n------\n')

    results['train_loss'].append(train_loss)
    results['train_acc'].append(train_acc)
    results['test_loss'].append(test_loss)
    results['test_acc'].append(test_acc)

end_time = timer()
total_time = end_time - start_time
print(f'Time to train: {total_time} seconds')

# Save the model
torch.save(model.state_dict(), 'models/cifar.pth')

# Plot the loss and accuracy curves over the training epochs
train_losses = torch.Tensor(results['train_loss']).cpu()
test_losses = torch.Tensor(results['test_loss']).cpu()
train_accs = torch.Tensor(results['train_acc']).cpu()
test_accs = torch.Tensor(results['test_acc']).cpu()
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.plot(range(epochs), train_losses, label = 'train loss')
plt.plot(range(epochs), test_losses, label = 'test_loss')
plt.title('loss curve')
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(epochs), train_accs, label = 'train accuracy')
plt.plot(range(epochs), test_accs, label = 'test accuracy')
plt.title('accuracy curve')
plt.legend()
plt.show()