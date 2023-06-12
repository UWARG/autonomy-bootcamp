"""
Contains functions for training and testing model.
"""
from typing import List, Tuple, Dict

import torch

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               device: torch.device) -> Tuple[float, float]:
    """
    Step through one epoch of training step

    Parameters
    ----------
    model: torch.nn.Module
        A PyTorch model to be trained.
    dataloader: torch.utils.data.DataLoader
        A DataLoader instance for the model to be trained on.
    loss_fn: torch.nn.Module
        A PyTorch loss function to minimize.
    optimizer: torch.optim.Optimizer
        A PyTorch optimizer to help minimize the loss function.
    device: torch.cuda.device
        A target device to compute on (e.g. "cuda" or "cpu").

    Returns
    -------
    Tuple[float, float]
        A tuple of training loss and training accuracy metrics. In the form (train_loss, train_accuracy).
    """
    train_loss, train_accuracy = 0, 0

    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        # Calculate and append loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_accuracy += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss /= len(data_loader)
    train_accuracy /= len(data_loader)
    return train_loss, train_accuracy

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module, 
              device: torch.cuda.device) -> Tuple[float, float]:
    """
    Step through one epoch of training step

    Parameters
    ----------
    model: torch.nn.Module
        A PyTorch model to be tested.
    dataloader: torch.utils.data.DataLoader
        A DataLoader instance for the model to be tested on.
    loss_fn: torch.nn.Module
        A PyTorch loss function to minimize.
    device: torch.cuda.device
        A target device to compute on (e.g. "cuda" or "cpu").

    Returns
    -------
    Tuple[float, float]
        A tuple of test loss and test accuracy metrics. In the form (test_loss, test_accuracy).
    """
    test_loss, test_accuracy = 0, 0

    model.eval()

    with torch.inference_mode():
        for X_test, y_test in data_loader:

            X_test, y_test = X_test.to(device), y_test.to(device)

            # forward pass
            test_pred_logits = model(X_test)

            # Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y_test)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_accuracy += ((test_pred_labels == y_test).sum().item()/len(test_pred_labels))

        test_loss /= len(data_loader)
        test_accuracy /= len(data_loader)
        return test_loss, test_accuracy

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List[float]]:
    """
    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Parameters
    ----------
    model: torch.nn.Module
        A PyTorch model to be trained and tested.
    train_loader: torch.utils.data.DataLoader
        A DataLoader instance for the model to be trained on.
    test_loader: torch.utils.data.DataLoader
        A DataLoader instance for the model to be tested on.
    optimizer: torch.optim.Optimizer
        A PyTorch optimizer to help minimize the loss function.
    loss_fn: torch.nn.Module
        A PyTorch loss function to calculate loss on both datasets.
    epochs: int
        An integer indicating how many epochs to train for.
    device: torch.device
        A target device to compute on

    Returns
    -------
    Dict[str, List]
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        {train_loss: [...],
        train_acc: [...],
        test_loss: [...],
        test_acc: [...]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model,
                                           data_loader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        data_loader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results
