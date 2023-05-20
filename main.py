"""
This is a starter file to get you going. You may also include other files if you feel it's necessary.

Make sure to follow the code convention described here:
https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions

Hints:
* The internet is your friend! Don't be afraid to search for tutorials/intros/etc.
* We suggest using a convolutional neural network.
* TensorFlow Keras has the CIFAR-10 dataset as a module, so you don't need to manually download and unpack it.
"""
import math

import torch 
from torch import nn
import torchvision
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

# GPU check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get training and validation data
train_data = torchvision.datasets.CIFAR10(
    root="cifar10_data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

val_data = torchvision.datasets.CIFAR10(
    root="cifar10_data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

# CIFAR10 labels
CLASS_LABELS = train_data.classes

# Make batches from the dataset
BATCH_SIZE = 32
train_dataloader = DataLoader(
    train_data,
    batch_size = BATCH_SIZE,
    shuffle = True,
)
val_dataloader = DataLoader(
    val_data,
    batch_size = BATCH_SIZE,
    shuffle = False,
)

# Model definition
class cifar10_model(nn.Module):
    def __init__(self):
        super().__init__()

        # Network
        self.network = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, (3,3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3,3), stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2),
            
            # Block 2
            nn.Conv2d(32, 64, (3,3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3), stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2),
        
            # Block 3
            nn.Conv2d(64, 128, (3,3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3,3), stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2),
    
            # Block 4
            nn.Flatten(),
            nn.Linear(in_features=4*4*128, out_features=128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
        )


    def forward(self, X):
        return self.network(X)
    
    def training_step(self, batch, criterion):
        """ 
        Calculates training loss for batch

        Returns tensor: loss
        """
        X_train, Y_train = batch
        X_train, Y_train = X_train.to(device), Y_train.to(device)

        # Forward prop
        Y_train_hat = self(X_train) 

        # Loss
        train_loss = criterion(Y_train_hat, Y_train)

        # Accuracy
        train_acc = self.acc_func(Y_train, Y_train_hat)

        return train_loss, train_acc
    
    # Validation step on entire validation set
    def validate(self, val_dataloader, criterion):
        self.eval()
        val_loss, val_acc = torch.Tensor([0]).to(device), torch.Tensor([0]).to(device)
        for batch in val_dataloader:
            X_val, Y_val = batch
            X_val, Y_val = X_val.to(device), Y_val.to(device)
            with torch.inference_mode():
                # Forward
                Y_val_hat = self(X_val)
                # Loss
                val_loss += criterion(Y_val_hat, Y_val)
                # Accuracy
                val_acc += self.acc_func(Y_val, Y_val_hat) 
        
        return val_loss, val_acc
    
    # Accuracy function
    def acc_func(self, truth, pred):
         _, prediction = torch.max(pred.data,1)
         return torch.tensor(torch.sum(prediction == truth).item() / len(prediction))

    def fit(self, train_dataloader, val_dataloader, epochs, optimizer, criterion):
        history = {
            'loss': torch.Tensor([]).to(device),
            'accuracy': torch.Tensor([]).to(device),
            'val_loss': torch.Tensor([]).to(device),
            'val_accuracy': torch.Tensor([]).to(device),
        }

        num_train_batches = len(train_dataloader)
        num_val_batches = len(val_dataloader)
        for epoch in range(epochs): 
            # Tracking performance
            epoch_loss = torch.Tensor([0]).to(device)
            epoch_acc = torch.Tensor([0]).to(device)

            # Training 
            self.train()
            for train_batch in train_dataloader:
                loss, acc = self.training_step(train_batch, criterion)
                epoch_loss = torch.add(epoch_loss, loss)
                epoch_acc = torch.add(epoch_acc, acc)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Validation 
            valid_loss, valid_acc = self.validate(val_dataloader, criterion)

            # Update history
            epoch_loss = epoch_loss / num_train_batches
            epoch_acc = epoch_acc / num_train_batches
            val_loss = valid_loss / num_val_batches
            val_acc = valid_acc / num_val_batches
            
            history['loss'] = torch.cat([history['loss'], epoch_loss])
            history['accuracy'] = torch.cat([history['accuracy'], epoch_acc])
            history['val_loss'] = torch.cat([history['val_loss'], val_loss])
            history['val_accuracy'] = torch.cat([history['val_accuracy'], val_acc])

            # Log performance 
            print(f'#Epoch {epoch+1}: loss = {epoch_loss.item():.6f}, acc = {epoch_acc.item():.6f}, val_loss = {val_loss.item():.4f}, val_accuracy = {val_acc.item():.4f}')
        
        return history
    

# Visualizer functions
def plot_loss(hist):
    loss = hist['loss'].to('cpu').detach().numpy()
    val_loss = hist['val_loss'].to('cpu').detach().numpy()
    plt.plot(loss, marker='o', label="Training Loss")
    plt.plot(val_loss, marker='*', label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_accuracy(hist):
    acc = hist['accuracy'].to('cpu').detach().numpy()
    val_acc = hist['val_accuracy'].to('cpu').detach().numpy()
    plt.plot(acc, marker='o', label="Training Accuracy")
    plt.plot(val_acc, marker='*', label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def infer_batch(model, batch, batch_size=BATCH_SIZE):
    X_test, Y_test = batch
    X, Y = X_test.to(device), Y_test.to(device)    
    Y_hat = model(X)

    X = X.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    Y = Y.to('cpu').detach().numpy()
    Y_hat = Y_hat.to('cpu').detach().numpy()

    fig, axes = plt.subplots(8,4, figsize=(15,30))
    for i in range(batch_size):
        x, y = math.floor(i/4), i % 4
        axes[x, y].imshow(X[i])
        axes[x, y].set_title(f'Actual: {CLASS_LABELS[Y[i]]} | Pred: {CLASS_LABELS[Y_hat[i].argmax()]}')
    plt.savefig(fname= r'\inference_results.png', format='png')
    plt.show()


# Testing model
my_model = cifar10_model()
my_model.to(device)

# Define hyperparams
optimizer = torch.optim.SGD(my_model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()
epochs = 12

# Train
history = my_model.fit(train_dataloader, val_dataloader, epochs, optimizer=optimizer, criterion=criterion)

# Visualize training
plot_loss(history)
plot_accuracy(history)

# Inference
infer_batch(my_model, next(iter(val_dataloader)))