import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

# checks if the gpu is available, otherwise uses cpu for training the model
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,),)])


# getting/storing the CIFAR10 dataset: train and test sets
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

# Loading the CIFAR10 Dataset into DataLoader
# we want to do this because iterating through the dataset itself gives us one sample at a time
# using the dataloader will allow us to pass samples in batches/chunks
# we set shuffle to True to reduce model overfitting (when the model has a high accuracy for identifying items on the trained set but low actual accuracy)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

class_names = ["airplane", "automobile", "bird", "cat",
               "deer", "dog", "frog", "horse", "ship", "truck"]


# defining the neural network by subclassing nn.Module
class NeuralNetwork(nn.Module):
    def __init__(self):  # this constructor initializes the neural network layers
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  # flattens the input data to a 1D array/tensor
        self.linear_relu_stack = nn.Sequential(  # sequence of layer operations -> data is passed through the modules in this order
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),  # non-linear activation function (just changes negative values to 0 - max(0, activation))
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),


            nn.Flatten(1, 3),  # converts tensor to 1d
            # # applies a linear transformation on the input using stored weight and biases
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
            nn.LogSoftmax(dim=1)
        )

        # note that the number of activations per layer is arbitrary - we need to modify to find an optimized model for the problem

    def forward(self, x):  # this is called when we pass the model input data -> don't call this directly
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


try:
    model = torch.load('model.pth')
    print("Loaded previous model")
except:
    model = NeuralNetwork().to(device)
    print("Generating a new model")


# ================= training the model =================

# loops through optimization code
def training_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (img, label) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(img)
        loss = loss_fn(pred, label)

        # Backpropagation: fine-tunes the weights for each layer based on the loss obtained per iteration
        optimizer.zero_grad()  # resets the gradients of the model parameters
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(img)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# evaluates the model's performance against test data


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for img, label in dataloader:
            pred = model(img)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss


# hyperparameters = let you control/fine-tune the optimization process
learning_rate = 0.001
batch_size = 5
epochs = 5


# initializing the loss function: used to measure the disparity between the obtained result of our model and the target value
# loss_fn normalizes the logits and computes the prediction error
loss_fn = nn.CrossEntropyLoss()

# initializing the optimizer: used to adjust model parameters to reduce error in each training step
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

loss_history = []

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    training_loop(train_dataloader, model, loss_fn, optimizer)
    valid_loss = test_loop(test_dataloader, model, loss_fn)

    loss_history.append(valid_loss)
print("Done!")

torch.save(model, "./model.pth")

x = [1, 2, 3, 4, 5]

plt.ylabel('loss')
plt.xlabel('epoch')
plt.xticks(ticks=x)
plt.plot(x, loss_history, marker=".", markersize="10")
plt.show()
