"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
from typing import List, Tuple

import torch
from torchvision import datasets, transforms

def data_loader(data_dir: str, batch_size: int, shuffle: bool=True) \
    -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, List[str]]:
    """
    Takes in a dataset directory and returns a train and test data loader

    Parameters
    ----------
    data_dir: str
        relative path to directory containing dataset
    batch_size: int
        batch size to load into torch.utils.data.DataLoader
    shuffle: bool
        whether to shuffle batches of torch.utils.data.DataLoader

    Retrns
    ------
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, List[str]]
        (train_dataloader, test_dataloader, class_names). class_names is a list of target classes.

    """
    normalize = transforms.Normalize(
        mean=[0.4799],
        std=[0.2386],
    )

    # define transforms
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        normalize
    ])

    # Load the dataset
    test_data = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )
    train_data = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader, class_names
