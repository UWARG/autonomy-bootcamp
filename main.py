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

import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
from time import sleep
from tqdm import tqdm

if __name__ == "__main__": # Here

    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    download_url(dataset_url, '.')

    with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
        tar.extractall(path='./data')

    data_dir = './data/cifar10'

    classes = os.listdir(data_dir + "/train")

    dataset = ImageFolder(data_dir+'/train', transform=ToTensor())

    img, label = dataset[0]
    img

    matplotlib.rcParams['figure.facecolor'] = '#ffffff'

    random_seed = 42
    torch.manual_seed(random_seed)

    val_size = 5000
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    len(train_ds), len(val_ds)

    batch_size=128

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

    def show_batch(dl):
        for images, labels in dl:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
            break

    def apply_kernel(image, kernel):
        ri, ci = image.shape       # image dimensions
        rk, ck = kernel.shape      # kernel dimensions
        ro, co = ri-rk+1, ci-ck+1  # output dimensions
        output = torch.zeros([ro, co])
        for i in range(ro): 
            for j in range(co):
                output[i,j] = torch.sum(image[i:i+rk,j:j+ck] * kernel)
        return output

    simple_model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(2, 2)
    )

    print("before imageclassification")
    
    class ImageClassificationBase(nn.Module):
        def training_step(self, batch):
            images, labels = batch 
            out = self(images)                  # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
            return loss
        
        def validation_step(self, batch):
            images, labels = batch 
            out = self(images)                    # Generate predictions
            loss = F.cross_entropy(out, labels)   # Calculate loss
            acc = accuracy(out, labels)           # Calculate accuracy
            return {'val_loss': loss.detach(), 'val_acc': acc}
            
        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        
        def epoch_end(self, epoch, result):
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss'], result['val_acc']))
            
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    print("before cifar")

    class Cifar10CnnModel(ImageClassificationBase):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

                nn.Flatten(), 
                nn.Linear(256*4*4, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 10))
            
        def forward(self, xb):
            return self.network(xb)

    model = Cifar10CnnModel()
    model

    print("get default device")

    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
        
    def to_device(data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list,tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    print("device data loader")

    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device
            
        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl: 
                yield to_device(b, self.device)

        def __len__(self):
            """Number of batches"""
            return len(self.dl)

    device = get_default_device()
    device

    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(model, device)

    print("start evaluate")

    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        for i in tqdm(range(10)):
            sleep(3)
        history = []
        optimizer = opt_func(model.parameters(), lr)
        for epoch in range(epochs):
            # Training Phase 
            model.train()
            train_losses = []
            for batch in tqdm(train_loader):
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # Validation phase
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)
        return history

    model = to_device(Cifar10CnnModel(), device)

    evaluate(model, val_dl)

    num_epochs = 10
    opt_func = torch.optim.Adam
    lr = 0.001

    print("start history")

    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

    def plot_accuracies(history):
        accuracies = [x['val_acc'] for x in history]
        plt.plot(accuracies, '-x')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs. No. of epochs')

    plot_accuracies(history)

    def plot_losses(history):
        train_losses = [x.get('train_loss') for x in history]
        val_losses = [x['val_loss'] for x in history]
        plt.plot(train_losses, '-bx')
        plt.plot(val_losses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs')

    plot_losses(history)

    test_dataset = ImageFolder(data_dir+'/test', transform=ToTensor())

    def predict_image(img, model):
        # Convert to a batch of 1
        xb = to_device(img.unsqueeze(0), device)
        # Get predictions from model
        yb = model(xb)
        # Pick index with highest probability
        _, preds  = torch.max(yb, dim=1)
        # Retrieve the class label
        return dataset.classes[preds[0].item()]

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in val_dl:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = Cifar10CnnModel(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)
    result = evaluate(model, test_loader)
    result

    torch.save(model.state_dict(), 'cifar10-cnn.pth')

    model2 = to_device(Cifar10CnnModel(), device)

    model2.load_state_dict(torch.load('cifar10-cnn.pth'))

    evaluate(model2, test_loader)
