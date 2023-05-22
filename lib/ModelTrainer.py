import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

class ModelTrainer():
    def __init__(
            self, 
            optimizer=None,
            loss_fn = None,
            epochs=3,
            batch_size=32,
            logging=True,
            graph_progress=True,
            seed=42
            ):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        if (epochs <= 0):
            raise Exception("Must have at least one epoch.")
        self.epochs = epochs
        if (batch_size <= 0):
            raise Exception("Must have batch size > 0.")
        self.batch_size = batch_size
        self.logging = bool(logging)
        self.graph_progress = graph_progress
        self.seed = seed
    
    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        return self

    def set_loss_function(self, loss_func: torch.nn.Module):
        self.loss_fn = loss_func
        return self
    
    def set_epochs(self, epochs: int):
        if (epochs <= 0):
            raise Exception("Must have at least one epoch.")
        self.epochs = epochs
        return self

    def set_batch_size(self, batch_size: int):
        if (batch_size <= 0):
            raise Exception("Must have batch size > 0.")
        self.batch_size = batch_size
        return self
    
    def set_logging(self, logging: bool):
        self.logging = logging
        return self

    def set_progress_graphing(self, graph_progress: bool):
        self.graph_progress = graph_progress
        return self
    
    def set_random_seed(self, seed: int):
        self.seed = seed
        return self
    
    def log(self, message):
        if (self.logging):
            print(message)
        
    # Calculate accuracy (a classification metric)
    @staticmethod
    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc

    def train(
            self, 
            model: torch.nn.Module, 
            training_data: torch.utils.data.DataLoader, 
            testing_data: torch.utils.data.DataLoader
            ):
        if (self.optimizer is None):
            raise Exception("Optimizer must be defined.")
        if (self.loss_fn is None):
            raise Exception("Loss function must be defined.")
          
        # set up graphing
        epoch_count = []
        training_accs = []
        testing_accs = []
        
        # set up device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.log(f"Using {device.type}")
        if (device.type == 'cuda'):
          model.cuda()
        
        # get dataloaders from datasets
        train_dataloader = DataLoader(
            training_data, 
            self.batch_size, 
            shuffle=True
        )
        test_dataloader = DataLoader(
            testing_data, 
            self.batch_size, 
            shuffle=False
        )
        model.train()
        
        # training loop
        for epoch in range(self.epochs):
            self.log(f"Epoch {epoch}")
            epoch_count.append(epoch)

            # training loss per batch
            train_loss, train_acc = 0,0

            

            for batch, (X, y) in enumerate(train_dataloader):
                # forward pass
                if device.type == 'cuda':
                  X, y = X.cuda(), y.cuda()
                y_pred = model(X)

                # calculate loss
                loss = self.loss_fn(y_pred, y)
                train_loss += loss.item()

                self.optimizer.zero_grad()
                # print(y,
                #     y_pred.argmax(dim=1))
                train_acc += ModelTrainer.accuracy_fn(
                    y_true=y,
                    y_pred=y_pred.argmax(dim=1)
                )

                # loss backward
                loss.backward()


                # optimizer step
                self.optimizer.step()

                if batch % 100 == 0:
                    self.log(f"Looked at batch {batch}, {batch * len(X)}/{len(train_dataloader.dataset)} samples")
            
            # calclate loss and accuracy per epoch
            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)
            self.log(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
            training_accs.append(train_acc)

            # test model
            test_loss, test_acc = 0,0
            model.eval()
            with torch.inference_mode():
                for X, y in test_dataloader:
                    X, y = X.cuda(), y.cuda()
                    
                    # forward pass
                    test_pred = model(X)

                    # calculate loss/accuracy
                    test_loss += self.loss_fn(test_pred, y)
                    test_acc += ModelTrainer.accuracy_fn(
                        y_true=y, 
                        y_pred=test_pred.argmax(dim=1)
                    )

                test_loss /= len(test_dataloader)
                test_acc /= len(test_dataloader)
            self.log(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
            testing_accs.append(test_acc)

        if self.graph_progress:
          plt.plot(epoch_count, training_accs, label="Training Accuracy")
          plt.plot(epoch_count, testing_accs, label="Testing Accuracy")
          plt.title("Training and Test Accuracy")
          plt.ylabel("Accuracy")
          plt.xlabel("Epochs")
          plt.legend()
          plt.show()
        return model