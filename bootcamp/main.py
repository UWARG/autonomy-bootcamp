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

from typing import Dict
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Your working code here

def main():
    """Main entry point to the program"""

    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'} device")

    training_data = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_data = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    labels_map: Dict[int, str] = {
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

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.show()

    print('lol')


# Excecute the main entry point
if __name__ == "__main__":
    main()
