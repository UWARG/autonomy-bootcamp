"""
This is a starter file to get you going. You may also include other files if you feel it's necessary.
Make sure to follow the code convention described here:
https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions
Hints:
* The internet is your friend! Don't be afraid to search for tutorials/intros/etc.
* We suggest using a convolutional neural network.
* TensorFlow Keras has the CIFAR-10 dataset as a module, so you don't need to manually download and unpack it.
"""

# Importing Libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch.nn as nn
import matplotlib.pyplot as plt

# Normalising testing and training data [-1,1], this ensures same scale is used for all data (faster convergence)
# Applying randomness noise with cropping and horizontal flipping to training transform
data_statistics = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
train_transforms_cifar = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # CxHxW
    transforms.Normalize(*data_statistics, inplace=True)  # [-1,1], data = (data-mean)/std_dev
])
test_transforms_cifar = transforms.Compose([
    transforms.ToTensor(),  # CxHxW
    transforms.Normalize(*data_statistics, inplace=True)  # [-1,1], data = (data-mean)/std_dev
])

# Storing the train and testing datasets, determined from 'bool train', applying transformations
dataset = torchvision.datasets.CIFAR10(root="data/", download=True, transform=train_transforms_cifar)
test_dataset = torchvision.datasets.CIFAR10(root="data/",
                                            download=True, train=False, transform=test_transforms_cifar)
# Creating datasets for training and validation
# The validation data will help eliminate data-memorizing, and will make up 20% of the training data
val_ratio = 0.2
train_dataset, val_dataset = random_split(dataset, [int((1-val_ratio)*len(dataset)), int(val_ratio * len(dataset))])

# Creating dataloading wrappers for the 3 datasets, used to access data more efficiently
batch_size = 32
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=True, pin_memory=True)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size, pin_memory=True)


# Determines whether the machine is Cuda enabled, allowing GPU access
def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")


# Sets data to the device passed as a parameter
def to_device(entity, device):
    if isinstance(entity, (list, tuple)):
        return [to_device(elem, device) for elem in entity]
    return entity.to(device, non_blocking=True) # non_blocking=True allows for asynchronous transfer


# Wraps around dataloader objects to transfer batches of data to specified device
class DeviceDataLoader:

    def __init__(self, dataloader, device):
        self.dl = dataloader
        self.device = device

    # Transfers batches of data to appropriate device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    # Returns number of total batches inside the dataloader
    def __len__(self):
        return len(self.dl)


# Getting the device to be used for training
device = get_default_device()

# Wrapping dataloaders with the device-data-transfer wrapper
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
test_dl = DeviceDataLoader(test_dl, device)


# Network architecture #

# Function which defines a convolutional block
def conv_block(in_channels, out_channels, pool=False):

    # Convolutional layer, followed by batch normalization and ReLu activation
    # Convolution layer used to extract features in the data
    # Batch normalization is used, over fitting is mitigated (more general to a collection of data)
    # Decreases internal covariate shift (where predecessing layers alter input data for future layers)
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),

              # This function removes linearity in the data
              # Includes sparsity as 0s are introduced. This keeps data more general
              # Reduces Vanishing Gradient moreso than sigmoid
              nn.ReLU(inplace=True)]

    # Pooling used to decrease spatial dimensions while maintaining important features
    # Helps identify general features with less weight on where in the input features are located
    if pool:
        layers.append(nn.MaxPool2d(2))

    return nn.Sequential(*layers)


# Class for the model architecture
class ResnetX(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # Includes two convolution layers followed by max pooling
        # These layers extract features, and are increased deeper in the network to include more complex patters
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)

        # Calling function which includes two more convolution layers
        # Residual layer used to decrease vanishing gradient
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        # Includes two convolution layers with both having pooling afterward
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)

        # Calling function which includes two more convolution layers
        self.res2= nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        # Linear layer which determines class matches
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    # Calls the layers in order
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)


# Creating an instance of the network
model = ResnetX(3, 10)


# training the model #

# Processing batch of data, determining how many of them were predicted correctly
def accuracy(logits, labels):
    pred, pred_class_id = torch.max(logits, dim=1) #BxN
    return torch.tensor(torch.sum(pred_class_id == labels).item() / len(logits))


# Creating Function to evaluate the accuracy and loss of the model
def evaluate(model, dl, loss_func):
    model.eval()
    batch_losses, batch_accs = [], []
    for images, labels in train_dl:
        with torch.no_grad():
            logits = model(images)
        batch_losses.append(loss_func(logits, labels))
        batch_accs.append(accuracy(logits, labels))
    epoch_avg_loss = torch.stack(batch_losses).mean().item()
    epoch_avg_acc = torch.stack(batch_accs).mean()
    return epoch_avg_loss, epoch_avg_acc


# Creating Function to train the network
def train(model, train_dl, val_dl, epochs, max_lr, loss_func, optim):
    # initialise the optimizer
    optimizer = optim(model.parameters(), max_lr)
    schedular = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs*len(train_dl))

    results = []

    # Iterating for the number of epochs
    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []

        # Iterating for each batch of data
        for images, labels in train_dl:

            # Storing output into logit
            logits = model(images)

            # Calculating loss of the model, between the output suggested class values and actual
            loss = loss_func(logits, labels)
            train_losses.append(loss)

            # Back-propogating through model, Calculating derivatives/gradients of the loss function
            # With respect to the parameters (weights and biases)
            loss.backward()  # delta_loss/ delta_model_parameters
            optimizer.step()
            optimizer.zero_grad()  # Must reset gradients so they are not accumulated

            # Storing learning rates and altering them
            lrs.append(optimizer.param_groups[0]["lr"])
            schedular.step()

        # Storing training loss
        epoch_train_loss = torch.stack(train_losses).mean().item()

        # Determining the accuracy of the validation model for fine-tuning
        epoch_avg_loss, epoch_avg_acc = evaluate(model, val_dl, loss_func)
        results.append({'avg_valid_loss': epoch_avg_loss, 'avg_valid_acc': epoch_avg_acc, 'avg_train_loss': epoch_train_loss, 'lr': lrs})
    return results


# Putting model onto appropriate device (GPU if Cuda enabled)
model = to_device(model, device)

# Defining various Model params, chosen after some guess and check
epochs = 16
max_lr = 1e-2

# Defining loss function and optimizer for reverse-gradient descent (decreasing loss function)
loss_func = nn.functional.cross_entropy
optim = torch.optim.Adam

# Training and storing results
results = train(model, train_dl, val_dl, epochs, max_lr, loss_func, optim)

# Printing the average accuracy for each epoch
for result in results:
    print(result["avg_valid_acc"])
print(" ")


# Plotting results
def plot(results, pairs):
    fig, axes = plt.subplots(len(pairs), figsize=(10, 10))
    for i, pair in enumerate(pairs):
        for title, graphs in pair.items():
            axes[i].set_title(title)
            axes[i].legend(graphs)
            for graph in graphs:
                axes[i].plot([result[graph] for result in results], '-x')


# Plotting the results
plot(results, [{"Accuraccies vs epochs": ["avg_valid_acc"]},
               {"Losses vs epochs": ["avg_valid_loss", "avg_train_loss"]}, {"Learning rates vs batches": ["lr"]}])
plt.show()

# Calculating the accuracy of the test batch
_, test_acc = evaluate(model, test_dl, loss_func)
print(test_acc)

# Saving the model
PATH = './cifar10-ResNetX.pth'
torch.save(model.state_dict(), PATH)

# Trying an untrained model for comparison
model2 = to_device(ResnetX(3,10), device)
_, test_acc = evaluate(model2, test_dl, loss_func)
print(test_acc)

# Reloading the saved model and retesting
model2.load_state_dict(torch.load("cifar10-ResNetX.pth"))
_, test_acc = evaluate(model2, test_dl, loss_func)
print(test_acc)
