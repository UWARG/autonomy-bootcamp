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

import os
import torch
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

#needed to use on Mac
if __name__ == "__main__":

    #importing dataset
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    download_url(dataset_url, '.')

    #extracting data from dataset
    with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
        tar.extractall(path='./data')

    #defining the directory with data
    data_dir = './data/cifar10'

    dataset = ImageFolder(data_dir+'/train', transform=ToTensor())
    
    #defining number of epochs and learning rate
    num_epochs = 10
    batch_size = 4
    opt_func = torch.optim.Adam
    learning_rate = 0.001
    
    #choosing device to use (gpu/cpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    val_size = 5000
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    len(train_ds), len(val_ds)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

    print("Implementing CNN")

    #convolutional neural network (CNN) implementation 
    class ImageClassificationBase(nn.Module):
        def training_step(self, batch):
            images, labels = batch 
            #generate predictions
            out = self(images)      
            #calculate loss
            loss = F.cross_entropy(out, labels)
            return loss
        
        def validation_step(self, batch):
            images, labels = batch 
            #generate predictions
            out = self(images)              
            #calculate loss
            loss = F.cross_entropy(out, labels)  
            #calculate accuracy
            acc = accuracy(out, labels)        
            return {'val_loss': loss.detach(), 'val_acc': acc}
            
        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            #combine losses
            epoch_loss = torch.stack(batch_losses).mean()  
            batch_accs = [x['val_acc'] for x in outputs]
            #combine accuracies
            epoch_acc = torch.stack(batch_accs).mean()   
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        
        def epoch_end(self, epoch, result):
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss'], result['val_acc']))
            
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    class Cifar10CnnModel(ImageClassificationBase):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

                nn.Flatten(), 
                nn.Linear(256*4*4, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 10))
            
        def forward(self, xb):
            return self.network(xb)

    model = Cifar10CnnModel()

    #using Cross Entropy Loss as loss metric
    criterion = nn.CrossEntropyLoss()
    #stochastic gradient descend 
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

    print("Beginning Training")

    #running CNN on training data and validation data and recording values for axis 
    n_total_steps = len(train_dl)

    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        history = []
        optimizer = opt_func(model.parameters(), lr)
        for epoch in range(epochs):
            # Training Phase 
            model.train()
            train_losses = []
            for batch in tqdm(train_loader):
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # Validation phase
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)
        return history

    history = fit(num_epochs, learning_rate, model, train_dl, val_dl, opt_func)    

    print('Finished Training')

    #plotting loss vs. no. of epochs on matplot (saving to folder)
    def plot_losses(history):
        train_losses = [x.get('train_loss') for x in history]
        val_losses = [x['val_loss'] for x in history]
        plt.plot(train_losses, '-bx')
        plt.plot(val_losses, '-rx')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs')
        plt.savefig("/users/jeessh/computer-vision-bootcamp/plot.png")
        
    plot_losses(history)

    #calculating accuracy by testing against validation set
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        for images, labels in val_dl:
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
