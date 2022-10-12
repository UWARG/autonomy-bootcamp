"""Class for running training and/or testing on neural networks."""


from functools import wraps
import logging
from time import perf_counter
from typing import Dict, List, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchsummary import summary

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
        LOG.info("Function %s: execution time=%s", f.__name__, te-ts)
        return ret
    return wrap

class Runner():
    """Class for running training and/or testing on neural networks."""
    def __init__(self, cmd: str, file: Optional[str] = None) -> None:

        self._cmd: str = cmd

        # Managers
        self.data_manager = DataManager()
        self.network = NeuralNetwork()

        # Training functions
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.network.parameters(),
                                   lr=0.001,
                                   momentum=0.9
                                   )

        self._path = file if file else "./models/bootcamp_nets.pth"
        self._training_metadata: Dict[str, List[float]] = {"loss": [], "accuracy": []}

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
        elif self._cmd == "summary":
            summary(self.network, (3, 32, 32))
        else:
            LOG.error("Invalid command: %s", self._cmd)

    @timing
    def __train(self) -> None:
        LOG.debug("Beginning training")

        for epoch in range(9):
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

            self._training_metadata["loss"].append(epoch_loss / len(self.data_manager.training_loader))
            self._training_metadata["accuracy"].append(correct_predictions * 100 / total_predictions)

            LOG.info("Epoch %d took %.3f seconds [Acc: %.3f%%, Loss: %.3f]", epoch + 1, perf_counter() - start_time,
                     self._training_metadata["accuracy"][epoch], self._training_metadata["loss"][epoch])


        torch.save(self.network.state_dict(), self._path)

        # Plot results
        _, (loss_plt, accuracy_plt) = plt.subplots(2)

        loss_plt.plot(self._training_metadata["loss"])
        loss_plt.set_title("Loss")
        accuracy_plt.plot(self._training_metadata["accuracy"])
        accuracy_plt.set_title("Accuracy")
        plt.tight_layout()

        plt.show()

        LOG.debug("Training complete")

    def __test(self) -> None:
        LOG.debug("Beginning testing")

        dataiter = iter(self.data_manager.test_loader)
        images, labels = dataiter.next()

        classes = self.data_manager.get_labels()

        LOG.info("Ground Truth: %s", ' '.join(f'{classes[labels[j]]:5s}' for j in range(self.data_manager.batch_size)))

        network = NeuralNetwork()
        network.load_state_dict(torch.load(self._path))

        outputs = network(images)

        _, predicted = torch.max(outputs, 1)

        LOG.info("Predicted: %s", ' '.join(f'{classes[predicted[j]]:5s}' for j in range(self.data_manager.batch_size)))

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

        LOG.info("Overall accuracy: %d %%", 100 * correct / total)