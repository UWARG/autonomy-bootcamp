"""
Used to create paths for saving data
"""
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms


# Global variables recognized as constants
# pylint: disable=[invalid-name]

# Determine when to save model
best_loss = 1000

# pylint: enable=[invalid-name]

# Constsants for training model and saving data
BATCH_SIZE = 128
NUM_EPOCHS = 1000

# Security measure to load C extensions not part of python
# pylint: disable=[no-member]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pylint: enable=[no-member]
MODEL_PATH = "./models"
PLOT_PATH = "./plots"


# Create paths for saving data if necessary
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(PLOT_PATH, exist_ok=True)


# Define transform for training data with data augmentation
# to reduce overfitting
transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
# Define transform for test data
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# Download train data and test data from PyTorch datasets
train_data = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
test_data = datasets.CIFAR10(
    root="./data", train=False, transform=transform_test, download=True
)

# Load data into DataLoader structure
train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Store statistics for plots
statistics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}


class Net(nn.Module):
    """
    Defines the CNN and forward function
    """

    def __init__(self):
        super().__init__()
        # Use nn.Sequential to simplify forward definition for CNN
        self.network = nn.Sequential(
            # Define CNN layers
            # First CONV > RELU > BATCHNORM > MAXPOOL layer
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # Second CONV > RELU > BATCHNORM > MAXPOOL layer
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.BatchNorm2d(50),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # Third CONV > RELU > BATCHNORM layer
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(100),
            # Dropout layer to reduce overfitting
            nn.Dropout(p=0.5, inplace=False),
            # Flatten to transition from convolution layer to fully connected layer
            nn.Flatten(),
            # LINEAR > RELU
            nn.Linear(in_features=900, out_features=500),
            nn.ReLU(),
            # Final classifier
            nn.Linear(in_features=500, out_features=10),
        )

    def forward(self, inp):
        """
        Defines forward function
        """
        out = self.network(inp)
        return out


model = Net()
model.to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

def save_model():
    """
    Function for saving a model to defined path
    """
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, "model.pth"))


def load_model():
    """
    Function for loading a model from defined path
    """
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "model.pth")))


def train_model(epoch):
    """
    Function for traing the model
    """

    print(f"EPOCH: {epoch}")

    # Put model into training mode
    model.train()

    # Define statistic variables
    train_loss = 0.0
    train_acc = 0.0
    train_correct = 0
    train_total = 0

    # Load data in batches with defined DataLoader
    for batch_num, data in enumerate(train_loader, 1):
        # Parse input and labels and send to device
        (images, labels) = data
        (images, labels) = (images.to(DEVICE), labels.to(DEVICE))

        # Zero out gradients
        optimizer.zero_grad()

        # Forward pass and compute loss
        pred = model(images)
        loss = loss_fn(pred, labels)

        # Backpropogation and weight updating
        loss.backward()
        optimizer.step()

        # Security measure to load C extensions not part of python
        # pylint: disable=[no-member]

        # Parse output, find class with greatest weight (prediction)
        _, prediction = torch.max(pred.data, 1)

        # pylint: enable=[no-member]

        # Update and calculate training staistics
        train_loss += loss.item()
        train_correct += (prediction == labels).sum().item()
        train_total += labels.size(0)
        train_acc = train_correct / train_total

        # Update printed training statistics after each batch
        print(
            f"Batch [{batch_num} / {len(train_loader)}] Training loss | "
            f"Training accuracy: {train_loss / batch_num:.4f} | "
            f"{100. * train_acc:.4f}%"
        )
        if batch_num != len(train_loader):
            print("\033[A\033[A")

    # Calculate, store and print training stats over entire epoch
    train_acc = train_correct / train_total
    train_loss = train_loss / len(train_loader)
    statistics["train_loss"].append(train_loss)
    statistics["train_acc"].append(train_acc)
    print("Epoch training loss: " + str(train_loss))
    print("Epoch training accuracy: " + str(train_acc))


def test_model():
    """
    Function for testing the model
    """

    # Put model into training mode
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    # Disable gradient calculation
    with torch.no_grad():
        # Load data in batches with defined DataLoader
        for batch_num, data in enumerate(test_loader, 1):
            # Parse input and labels and send to device
            (images, labels) = data
            (images, labels) = (images.to(DEVICE), labels.to(DEVICE))

            # Forward pass and compute loss
            pred = model(images)
            loss = loss_fn(pred, labels)

            # Security measure to load C extensions not part of python
            # pylint: disable=[no-member]

            # Parse output, find class with greatest weight (prediction)
            _, prediction = torch.max(pred.data, 1)

            # pylint: enable=[no-member]

            # Update and testing staistics
            test_loss += loss.item()
            test_correct += (prediction == labels).sum().item()
            test_total += labels.size(0)
            test_acc = test_correct / test_total

            # Update printed testing statistics after each batch
            print(
                f"Batch [{batch_num} / {len(test_loader)}] Validation loss | "
                f"Validation accuracy: {test_loss / batch_num:.4f} | "
                f"{100. * test_acc:.4f}%"
            )
            if batch_num != len(test_loader):
                print("\033[A\033[A")

    # Calculate, store and print training stats over entire epoch
    test_acc = test_correct / test_total
    test_loss = test_loss / len(test_loader)
    statistics["val_loss"].append(test_loss)
    statistics["val_acc"].append(test_acc)
    print("Epoch validation loss: " + str(test_loss))
    print("Epoch validation accuracy: " + str(test_acc))


def plot_model(epoch):
    """
    Function for plotting the model
    """
    plt.style.use("ggplot")

    # Plot and save accuracy staistics
    plt.figure()
    plt.plot(range(1, epoch + 1), statistics["val_acc"], label="val_acc")
    plt.plot(
        range(1, epoch + 1), statistics["train_acc"], label="train_acc"
    )
    plt.title("Training and Validation Accuracy on CIFAR-10 Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(PLOT_PATH, "acc_plot.png"))
    plt.close()

    # Plot and save loss staistics
    plt.figure()
    plt.plot(range(1, epoch + 1), statistics["val_loss"], marker="o", label="val_loss")
    plt.plot(
        range(1, epoch + 1),
        statistics["train_loss"],
        marker="o",
        label="train_loss",
    )
    plt.title("Training and Validation Loss on CIFAR-10 Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(PLOT_PATH, "loss_plot.png"))
    plt.close()


# Train, test and plot the model over desired number of epochs
# Save if necessary
for e in range(0, NUM_EPOCHS):
    train_model(e + 1)
    test_model()
    if statistics["val_loss"][-1] < best_loss:
        print("Saving model")
        save_model()
        best_loss = statistics["val_loss"][-1]
    plot_model(e + 1)

    # Update learning rate
    scheduler.step()
