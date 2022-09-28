"""Class for running training and/or testing on neural networks."""


from functools import wraps
import logging
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


LOG = logging.getLogger(__name__)

# Function timer decorator
def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = perf_counter()
        ret = f(*args, **kwargs)
        te = perf_counter()
        LOG.debug("Function %s: execution time=%s", f.__name__, te-ts)
        return ret
    return wrap

class Runner():
    """Class for running training and/or testing on neural networks."""
    def __init__(self, cmd: str, file: Optional[str] = None) -> None:

        self._cmd: str = cmd
        self._file: Optional[str] = file

        # Managers
        self.data_manager = DataManager()
        self.network = NeuralNetwork()

        # Training functions
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.network.parameters(),
                                   lr=0.004,
                                   momentum=0.95
                                   )

        self.path = "models/bootcamp_nets.pth"

    def run(self) -> None:
        LOG.info("Using %s device", "cuda" if torch.cuda.is_available() else "cpu")

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

        if self._cmd == "train":
            self.__train()
        elif self._cmd == "test":
            self.__test()
        else:
            LOG.error("Invalid command: %s", self._cmd)

    @timing
    def __train(self) -> None:
        LOG.debug("Beginning training")

        for epoch in range(20):
            start_time = perf_counter()

            running_loss: float = 0.0
            epoch_loss: float = 0.0

            correct_predictions: int = 0
            total_predictions: int = 0

            for i, data in enumerate(self.data_manager.training_loader, 0):
                # Get the inputs and labels from the data
                inputs, labels = data

                # Reset optimizer gradients
                self.optimizer.zero_grad()

                outputs = self.network(inputs)

                predictions: torch.Tensor = torch.argmax(outputs, dim=1)  # Run predictions
                correct_predictions += (predictions == labels).sum().item()  # Count correct predictions
                total_predictions += labels.size(0)  # Count total predictions

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Stats
                running_loss += loss.item()
                epoch_loss += loss.item()
                if i % 2000 == 0:
                    LOG.debug('Epoch: %d, Batch: %5d, loss: %.3f', epoch + 1, i + 1, running_loss / 2000)
                    running_loss = 0.0

            LOG.info("Epoch %d took %s seconds [Acc: %.3f%%, Loss: %.3f]", epoch + 1, perf_counter() - start_time,
                     correct_predictions * 100 / total_predictions, epoch_loss / len(self.data_manager.training_loader))

        torch.save(self.network.state_dict(), self.path)

        LOG.debug("Training complete")

    def __test(self) -> None:
        LOG.debug("Beginning testing")

        dataiter = iter(self.data_manager.test_loader)
        images, labels = dataiter.next()

        classes = self.data_manager.get_labels()

        def imshow(img):
            img = img / 2 + 0.5     # unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

        # print images
        imshow(torchvision.utils.make_grid(images))
        LOG.info("Ground Truth: %s", ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

        network = NeuralNetwork()
        network.load_state_dict(torch.load(self.path))

        outputs = network(images)

        _, predicted = torch.max(outputs, 1)

        LOG.info("Predicted: %s", ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.data_manager.test_loader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = network(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        LOG.info("Accuracy of the network on the 10000 test images: %d %%", 100 * correct / total)