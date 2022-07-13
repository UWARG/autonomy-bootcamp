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



import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from model import Net

# Your working code here

# to check if CUDA is available to train on GPU if possible (if not, train on CPU)
def check_gpu():
    global train_on_gpu
    if torch.cuda.is_available():
        train_on_gpu = True
        cuda_available = True
        print('CUDA available! Training on GPU...')
    else:
        train_on_gpu = False
        cuda_available = False
        print('CUDA not available. Training on CPU...')
    return train_on_gpu, cuda_available

def mainconstant():
    num_workers = 0
    batch_size = 20
    valid_size = 0.2
    return num_workers, batch_size, valid_size


# function to convert data to a normalized torch.FloatTensor i.e. a tensor with values between 0 and 1

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load the training and test datasets
def load_data():
    train_data = datasets.CIFAR10('data', train=True,
                                  download=True, transform=transform)


    test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)
    return train_data, test_data


# obtain training indices that will be used for validation set
def get_valid_idx(train_data):
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    valid_size = 0.2
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    return train_idx, valid_idx

# defining samplers for obtaining training, validation and test batches
def get_train_valid_loaders(train_data, test_data, train_idx, valid_idx, batch_size, num_workers):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(train_idx), num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(valid_idx), num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader


# specifying the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image








# create a complete CNN
def create_model():
    model = Net()
    print(model)
    return model

model = create_model()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)



# number of epochs to train the model
def constants():
    n_epochs = 25
    epoch_list = range(1, 26)
    train_losslist = []
    valid_loss_min = np.Inf
    return n_epochs, epoch_list, train_losslist, valid_loss_min


def train_model(model, train_loader, valid_loader, n_epochs):
    n_epochs = 25
    epoch_list = range(1, 26)
    train_losslist = []
    valid_loss_min = np.Inf
    for epoch in range(1, n_epochs + 1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
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
            train_loss += loss.item()*data.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss += loss.item()*data.size(0)

        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)
        train_losslist.append(train_loss)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'model_cifar.pt')
            valid_loss_min = valid_loss

# helper function to plot the loss over the epochs
def plot_loss(epoch_list, train_losslist, valid_loss_min):
    plt.plot(epoch_list, train_losslist, 'b', label='Training loss')
    plt.plot(epoch_list, valid_loss_min, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

#helper function to load the model
def load_model(model, model_name):
    model.load_state_dict(torch.load('model_cifar.pt'))
    return model





#helper function to iterate over test data

def test_model(model, test_loader):
    global test_loss
    global class_correct
    global class_total
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    model.eval()
    for data, target in test_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
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
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    # average test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    




#function to calculate test accuracy
def test_accuracy(class_correct, class_total):
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



# main function to run the model
def main():
    num_workers, batch_size, valid_size = mainconstant()
    n_epochs, epoch_list, train_losslist, valid_loss_min = constants()
    check_gpu()

    # load the data
    train_data, test_data = load_data()
    train_idx, valid_idx = get_valid_idx(train_data=train_data)
    train_loader, valid_loader, test_loader =get_train_valid_loaders(train_data=train_data, test_data=test_data, train_idx=train_idx, valid_idx=valid_idx, batch_size=batch_size, num_workers=num_workers)
    # build the model
    model = create_model()
    train_model(model, train_loader, valid_loader, n_epochs=n_epochs)
    test_model(model, test_loader)
    test_accuracy(class_correct, class_total)
    plot_loss(epoch_list, train_losslist, valid_loss_min)
    new_model = load_model(model, 'model_cifar.pt')
    test_model(new_model, test_loader)
    test_accuracy(class_correct, class_total)
    plot_loss(epoch_list, train_losslist, valid_loss_min)



main()