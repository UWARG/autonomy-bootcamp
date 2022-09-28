"""Class for running training and/or testing on neural networks."""


from functools import wraps
from time import perf_counter
from typing import Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision

import matplotlib.pyplot as plt


from bootcamp.data_manager import DataManager
from bootcamp.neural_network import NeuralNetwork

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

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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