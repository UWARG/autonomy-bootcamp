# imports
from cmath import inf
import matplotlib.pyplot as plt
from model import Net
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms

# packaging constants into one class
class Constants:
    def __init__(self, num_workers, batch_size, valid_size, n_epochs, train_losses, valid_losses, valid_loss_min, train_loss, valid_loss):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.n_epochs = n_epochs
        self.train_losses = train_losses
        self.valid_losses = valid_losses
        self.valid_loss_min = valid_loss_min
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.epoch_list = range(1, self.n_epochs + 1)
    
# data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# get training and validation datasets
def load_data():
    train_data = datasets.CIFAR10('data', train=True,
                                download=True, transform=transform)
    test_data = datasets.CIFAR10('data', train=False,
                                download=True, transform=transform)
    return train_data, test_data

# training indices for validation
def get_indices(constants, train_data):
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(constants.valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    return train_idx, valid_idx

def get_loaders(constants, train_data, test_data, train_idx, valid_idx):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=constants.batch_size,
        sampler=SubsetRandomSampler(train_idx), num_workers=constants.num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=constants.batch_size, 
        sampler=SubsetRandomSampler(valid_idx), num_workers=constants.num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=constants.batch_size, 
        num_workers=constants.num_workers)
    return train_loader, valid_loader, test_loader

# image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# create nn and move tensors to GPU if CUDA is available
def create_model():
    model = Net()
    global criterion, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    if torch.cuda.is_available():
        model.cuda()
    return model

def train_model(model, train_loader, valid_loader, constants):
    for epoch in range(1, constants.n_epochs+1):

        constants.train_loss = 0.0
        constants.valid_loss = 0.0
       
        model.train()
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            constants.train_loss += loss.item()*data.size(0)
            
        # validate the model
        
        model.eval()
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            constants.valid_loss += loss.item()*data.size(0)
        
        # calculate average losses
        constants.train_loss/=len(train_loader.dataset)
        constants.valid_loss/=len(valid_loader.dataset)
        
        # At completion of epoch
        constants.train_losses.append(constants.train_loss)
        constants.valid_losses.append(constants.valid_loss)
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.10f} \tValidation Loss: {:.10f}'.format(
            epoch, constants.train_loss, constants.valid_loss))
        
        # save model if validation loss has decreased
        if constants.valid_loss <= constants.valid_loss_min:
            print('Validation loss decreased ({:.10f} --> {:.10f}).  Saving model ...'.format(
            constants.valid_loss_min,
            constants.valid_loss))
            torch.save(model.state_dict(), 'model_cifar.pt')
            constants.valid_loss_min = constants.valid_loss

def load_model(model):
    model.load_state_dict(torch.load('model_cifar.pt'))

def test_model(model, test_loader, constants):
    # track test loss
    global test_loss, class_correct, class_total
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()
    # iterate over test data
    for data, target in test_loader:
        # move tensors to GPU if CUDA is available
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(constants.batch_size): 
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

def print_test_accuracy(class_correct, class_total, test_loader, test_loss):
    # average test loss
    test_loss /= len(test_loader.dataset)
    print('\nTest Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

def plot_loss(constants):
    plt.plot(constants.epoch_list, constants.train_losses, label='Training loss')
    plt.plot(constants.epoch_list, constants.valid_losses, label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(frameon=False)
    plt.show()

def main():
    constants = Constants(num_workers=0, batch_size=20, valid_size=0.2, n_epochs=30, train_losses=[], valid_losses=[], valid_loss_min=np.inf, train_loss=0.0, valid_loss=0.0)

    # loading the data
    train_data, test_data = load_data()
    train_idx, valid_idx = get_indices(constants=constants, train_data=train_data)
    train_loader, valid_loader, test_loader = get_loaders(train_data=train_data, test_data=test_data, train_idx=train_idx, valid_idx=valid_idx, constants=constants)
    
    # creating model
    model = create_model()

    # training model
    train_model(model=model, train_loader=train_loader, valid_loader=valid_loader, constants=constants)

    # testing model
    test_model(constants=constants, model=model, test_loader=test_loader)
    print_test_accuracy(class_correct=class_correct, class_total=class_total, test_loader=test_loader, test_loss=test_loss)

    # loading best trained model
    best_model = load_model(model=model)

    # plotting losses
    plot_loss(constants=constants)


main()

