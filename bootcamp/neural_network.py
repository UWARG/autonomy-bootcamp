"""Class defining the structure and handling of a neural network"""

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """Class defining the structure and handling of a neural network"""

    def __init__(self) -> None:
        super().__init__()

        # Make sure that padding = kernel_size // 2 to keep the same image size
        self.kernel_size: int = 3
        self.padding: int = 1
        # 3 and 1 work best out of the pairs (3, 1), (5, 2), (7, 3)

        def conv_layer(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding),
                nn.ReLU(),
                # I need to learn more about batch normalization, but adding it greatly speeds up training
                nn.BatchNorm2d(out_channels),
                # Pooling after each convolution helps reduce the number of calculations while maintaining accuracy
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        def linear_layers() -> nn.Sequential:
            return nn.Sequential(
                nn.Flatten(), # Flatten all dimensions except batch
                nn.Linear(256, 100),
                nn.ReLU(),
                nn.Linear(100, 20),
                nn.ReLU(),
                nn.Linear(20, 10)
            )

        # Network architecture consists of layers and functions
        self.network = nn.Sequential(
            conv_layer(3, 16),
            conv_layer(16, 32),
            conv_layer(32, 64),
            conv_layer(64, 128),
            conv_layer(128, 256),
            linear_layers()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Steps the training forwards through the network architecture

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Most likely classificiation
        """
        return self.network(x)
