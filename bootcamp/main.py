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

import argparse
from functools import wraps
from time import perf_counter
from typing import Dict, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Your working code here

# Function timer decorator
def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = perf_counter()
        ret = f(*args, **kwargs)
        te = perf_counter()
        print(f"Function {f.__name__} execution time: {te-ts}s")
        return ret
    return wrap

class NeuralNetwork(nn.Module):
    """Class defining the structure and handling of a neural network"""

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Steps the training forwards

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class DataManager():
    """Class for managing the datasets for the neural networks."""
    def __init__(self) -> None:
        self.batch_size: int = 4
        # Load data and labels
        # These shouldn't be accessed directly, but I'm running out of time, and I just need to finish this
        self.training_data = CIFAR10(root="data",
                                     train=True,
                                     download=True,
                                     transform=transforms.ToTensor())  # Only basic transform to tensor is needed
        self.training_loader = DataLoader(dataset=self.training_data,
                                          batch_size=self.batch_size,
                                          shuffle=True)

        self.test_data = CIFAR10(root="data",
                                 train=False,
                                 download=True,
                                 transform=transforms.ToTensor())
        self.test_loader = DataLoader(dataset=self.test_data,
                                          batch_size=self.batch_size,
                                          shuffle=True)

        self.labels_map: Dict[int, str] = {
            0: "Plane",
            1: "Car",
            2: "Bird",
            3: "Cat",
            4: "Deer",
            5: "Dog",
            6: "Frog",
            7: "Horse",
            8: "Ship",
            9: "Truck",
        }

class Runner():
    """Class for running training and/or testing on neural networks."""
    def __init__(self, cmd: str, file: Optional[str] = None) -> None:

        self._cmd: str = cmd
        self._file: Optional[str] = file

        # Managers
        self.dataset = DataManager()
        self.network = NeuralNetwork()

        # Training functions
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.network.parameters(),
                                   lr=0.001,
                                   momentum=0.9
                                   )

        self.path = "./bootcamp_nets.pth"

    def run(self) -> None:
        print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'} device")

        # figure = plt.figure(figsize=(8, 8))
        # cols, rows = 3, 3
        # for i in range(1, cols * rows + 1):
        #     sample_idx = torch.randint(len(self.dataset.training_data), size=(1,)).item()
        #     img, label = self.dataset.training_data[sample_idx]
        #     figure.add_subplot(rows, cols, i)
        #     plt.title(self.dataset.labels_map[label])
        #     plt.axis("off")
        #     plt.imshow(img.permute(1, 2, 0).squeeze(), cmap="gray")
        # plt.show()

        self.__train()
        self.__test()

    @timing
    def __train(self) -> None:
        for epoch in range(2):

            running_loss: float = 0.0
            for i, data in enumerate(self.dataset.training_loader, 0):
                # Get the inputs and labels from the data
                inputs, labels = data

                # Reset optimizer gradients
                self.optimizer.zero_grad()

                outputs = self.network(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Stats
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        torch.save(self.network.state_dict(), self.path)

        print('Training Done')

    def __test(self) -> None:
        dataiter = iter(self.dataset.test_loader)
        images, labels = dataiter.next()

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        def imshow(img):
            img = img / 2 + 0.5     # unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

        # print images
        imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

        network = NeuralNetwork()
        network.load_state_dict(torch.load(self.path))

        outputs = network(images)

        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                    for j in range(4)))

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.dataset.test_loader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = network(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


if __name__ == "__main__":
    """Starts the bootcamp program and parses the arguments."""

    parser = argparse.ArgumentParser()
    cmd_help_str: str = """Command to run. 'train' will train a new neural network and store it in a new file, test will
                        load a neural network from the latest file."""

    parser.add_argument("command", help=cmd_help_str, choices=["train", "test"])

    parser.add_argument("-f", "--file", help="File to load or save the neural network to/from.", type=str)

    args = parser.parse_args()
    runnner = Runner(args.command, args.file)


