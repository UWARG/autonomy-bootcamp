from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.utils import makegrid


BATCH_SIZE = 8
NUM_EPOCHS = 15
LEARNING_RATE = 0.01
#Taking 2000 images for validation, the rest is for training
VALIDATION_SIZE = 2000


def show_img(img):
    """
    Unnormalizes an tensor, converts it to a numpy array and displays it

    Parameters
    ----------
    img: tensor
        The image to display
    
    """
    img = img / 2 + 0.5 
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def accuracy(outputs, labels):
    """
    Calculates model accuracy in validation tests

    Parameters
    ----------
    outputs: Torch Tensor
        Output from the model when guessing the class of an image
    labels: Torch Tensor
        Labels of a batch of images

    Return
    ----------
    Returns a tensor containing the average accuracy guessing a batch
    """
    _, prediction = torch.max(outputs,1)
    return torch.tensor(torch.sum(prediction == labels).item() / len(prediction))

  

def evaluate(model, val_loader):
    """
    Evaluates the model validation loss and accuracy over an epoch

    Parameters
    ----------
    model: Torch Neural Network
        The model that is being evaluated
    val_loader: Torch Tensor
        The validation set loader

    Return
    ----------
    Returns a dictionary containing the mean of the validation loss and validation accuracy
    over an epoch
    """
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)



class Net(nn.Module):
    """
    The Convolution Neural Network

    Method
    ----------
    __init__()
        Sets initial neural network layers
    forward()
        Performs the model's forward pass
    training_step()
        Calculates training loss for a batch
    validation_step()
        Calculates Validation loss and accuracy for a batch
    validation_epoch_end()
        Calculates validation loss and accuracy mean for an epoch
    epoch_end()
        Outputs training loss, validation loss, and validation accuracy statistics 
        at the end of an epoch
    
    """
    def __init__(self):
        super().__init__()
        """
        Network constructor
        """

        self.network = nn.Sequential(
            
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Flatten(),
        nn.Linear(4096, 120),
        nn.ReLU(),
        nn.Linear(120, 80),
        nn.ReLU(),
        nn.Linear(80, 10)
        )
    
    def forward(self, x):
        return self.network(x)
    

    def training_step(self, batch):
        """
        Calculates the training loss for a batch

        Parameters:
        ----------
        self: Net class
            Implicitly passed neural network
        batch: Torch Tensor
            Current Batch

        Return
        ----------
        Returns the training loss for that batch
        """
        images, labels = batch 
        outputs = self(images) 
        loss = criterion(outputs, labels) 
        return loss
    
 
    def validation_step(self, batch):
        """
        Calculates the validation loss and accuracy for a batch

        Parameters:
        ----------
        self: Net class
            Implicitly passed neural network
        batch: Torch Tensor
            Current Batch

        Return
        ----------
        Returns a dictionary containing validation loss for that batch and the validation accuracy
        """
        images, labels = batch 
        outputs = self(images) 
        
        val_loss = criterion(outputs, labels) 
        acc = accuracy(outputs, labels)          
        return {'val_loss': val_loss.detach(), 'val_acc': acc}
    

    def validation_epoch_end(self, outputs):
        """
        Calculates the validation loss and accuracy for an epoch

        Parameters:
        ----------
        self: Net class
            Implicitly passed neural network
        batch: Torch Tensor
            Current Batch

        Return
        ----------
        Returns a dictionary containing the validation loss and accuracy mean for an epoch
        """
        batch_losses = []
        batch_accuracies = []
        
        for output in outputs:
            batch_losses.append(output['val_loss'])
            batch_accuracies.append(output['val_acc'])
        
        epoch_loss = torch.stack(batch_losses).mean() 
        epoch_acc = torch.stack(batch_accuracies).mean() 
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    
    def epoch_end(self, epoch, result):
        """
        Calculates the training loss for a batch

        Parameters:
        ----------
        self: Net class
            Implicitly passed neural network
        epoch: int
            The current epoch number
        result: dictionary
            Results from an epoch

        Return
        ----------
        Returns the training loss for that batch
        """
        current_loss = result["train_loss"]
        current_val_acc =  result["val_acc"]
        print(f'Epoch [{epoch+1}], training loss: {result["train_loss"]:.3f}, validation loss: {result["val_loss"]:.4f}, validation accuracy: {result["val_acc"]:.4f}')



def fit(epochs, lr, model, training_loader, validation_loader):
    """
        Fits a model

        Parameters:
        ----------
        epochs: int
            The amount of epochs to iterate through for training
        lr: float
            The learning rate of the model
        model: Net class
            The model that is being trained
        training_loader: Torch Dataloader
            The dataloader containing training images
        validation_loader: Torch Dataloader
            The dataloader containing validation images

        Return
        ----------
        Returns a list containing dictionaries of the training loss, validation loss, and
        validation accuracy for each epoch 
        """
    history = []
    optimizer = torch.optim.SGD(model.parameters(),lr, momentum = 0.9)
    for epoch in range(epochs):
        
        model.train()
        train_losses = []
        for batch in training_loader:

            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        result = evaluate(model, validation_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    return history



def plot_losses(history):
    """
        Plots the training losses against validation losses after training 

        Parameters:
        ----------
        history: list
            list containing dictionaries of the training loss, validation loss, and 
            validation accuracies of each epoch

        """
    train_losses = []
    val_losses = []
    
    for epoch_vals in history:
        train_losses.append(epoch_vals.get('train_loss'))
        val_losses.append(epoch_vals['val_loss'])

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Training Loss and Validation Loss Through Each Epoch')



def plot_accuracies(history):
    """
    Plots the validation accuracy of each epoch

    Parameters:
    ----------
    history: list
        list containing dictionaries of the training loss, validation loss, and 
        validation accuracies of each epoch

    """
    accuracies = []
    for epoch_vals in history:
        accuracies.append(epoch_vals['val_acc'])

    plt.plot(accuracies, '-x')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs. No. of epochs');


def test_model(model, test_loader, y_predicted, y_actual):
    """
        Tests a model

        Parameters:
        ----------
        model: Net class
            The model being tested
        test_loader: Torch Dataloader
            The dataloader containing test images
        y_predicted: list
            List to fill with the predicted classes of the images           
        y_actual: list
            List to fill with actual classes of the images

        """
    with torch.no_grad():
    for data in test_loader:
        images, labels = data
        
        # calculate outputs by running images through the network
        outputs = model(images)
        outputs_heat = model(images)
        
        #counting the amount of images seen and the amount that is correct and iterating
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        
        outputs_heat = (torch.max(torch.exp(outputs_heat), 1)[1]).data.cpu().numpy()
        y_predicted.extend(outputs_heat) # predicted values
        
        labels_heat = labels.data.cpu().numpy()
        y_actual.extend(labels_heat) # actual values

    print(f'Accuracy of the network on the 10000 test images: {100*correct//total} %')


def heatmap(y_predicted, y_actual):
    """
        Creates a heatmap of all class predictions

        Parameters:
        ----------
        y_predicted: list
            List containing the predicted classes from a model
        y_actual: list
            List containing the actual classes of images

        """
    #setting a confusion matrix for a heatmap
    cf_matrix = confusion_matrix(y_actual, y_predicted)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    plt.title("Heatmap of model accuracy")
    sns.heatmap(df_cm, annot=True)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

training_data_raw = CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testing_data = CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_size = len(training_data_raw) - val_size 

#splitting the raw 50000 training data into a training and validation data set
training_data, val_data = random_split(training_data_raw, [train_size, VALIDATION_SIZE])

#confirming a proper split
print(f"Length of Training Data : {len(training_data)}")
print(f"Length of Validation Data : {len(val_data)}")
print(f"Length of test Data : {len(testing_data)}")

#Making loaders for all data
train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

#Printing Class Names
classes = training_data_raw.classes
print(classes)
print(type(training_data_raw))


# get some random training images
data_iter = iter(train_loader)
images, labels = next(data_iter)
print(labels)

#showing the images and the labels in the same order
show_img(make_grid(images))
for img_num in range(batch_size):
    print(classes[labels[img_num]])

plt.show()

net = Net()
opt_func = torch.optim.Adam
criterion = nn.CrossEntropyLoss()

#fitting the model on training data and record the result after each epoch
history = fit(NUM_EPOCHS, LEARNING_RATE, net, train_loader, val_loader, opt_func)

#saving the model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

data_iter = iter(test_loader)
images, labels = next(data_iter)

# print images
show_img(torchvision.utils.make_grid(images))
outputs = net(images)
_, predicted = torch.max(outputs, 1)

#outputting the actual and predicted classes of each image
for pred_img in range(8):
    print("Actual: ", ''.join(f'{classes[labels[pred_img]]:10s}'),        
          "   Predicted: ",''.join(f'{classes[predicted[pred_img]]:5s}'))

plot_losses(history)
plot_accuracies(history)

y_pred = []
y_true = []

test_model(net, test_loader, y_pred, y_true)
heatmap(y_pred, y_true)
#accuracy of 75 %
