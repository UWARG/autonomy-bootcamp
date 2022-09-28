"""Class for managing the datasets for the neural networks."""

from typing import Dict
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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