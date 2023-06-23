# Autonomy Bootcamp Submission

This program trains a convolutional neural net on the CIFAR10 dataset.
> The PEP 8 and Pylint styles are used to format this code.

## Training and Test Loss and Accuracy Curves

<img src="loss_curves.PNG" width="500"> <img src="accuracy_curves.PNG" width="487">

### Final accuracy achieved
![] ()

### Program output
```
CUDA is not available.  Training on CPU ...

Files already downloaded and verified
Files already downloaded and verified

CUDA is not available.  Training on CPU ...

Start Training

[Epoch: 1]
Training loss: 1.822
Train accuracy: 44.146
----------
Test loss: 1.532
Test accuracy: 43.58
----------

[Epoch: 2]
Training loss: 1.409
Train accuracy: 53.96
----------
Test loss: 1.300
Test accuracy: 52.91
----------

[Epoch: 3]
Training loss: 1.248
Train accuracy: 56.856
----------
Test loss: 1.256
Test accuracy: 54.97
----------

[Epoch: 4]
Training loss: 1.151
Train accuracy: 62.828
----------
Test loss: 1.160
Test accuracy: 59.03
----------

[Epoch: 5]
Training loss: 1.075
Train accuracy: 65.368
----------
Test loss: 1.101
Test accuracy: 61.2
----------

[Epoch: 6]
Training loss: 1.009
Train accuracy: 67.426
----------
Test loss: 1.071
Test accuracy: 62.5
----------

[Epoch: 7]
Training loss: 0.953
Train accuracy: 68.922
----------
Test loss: 1.074
Test accuracy: 62.16
----------

Finished Training

Accuracy of the network on the 10000 test images: 61 %

Accuracy for each class:

plane: 65.0 %
car  : 70.6 %
bird : 55.0 %
cat  : 44.0 %
deer : 62.0 %
dog  : 49.5 %
frog : 71.1 %
horse: 60.3 %
ship : 76.7 %
truck: 62.0 %

Device: cpu
```

## Training process
__Attempt 1:__
```
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
<img src="accuracy_attempt1.png" width="500"> 
<img src="class_accuracy_attempt1.png" width="500"> 
** include loss plots and time

This model gives an accuracy of 62%. Evidently, there is room for improvement.
In the next few attempts, the model and the hyperparameters are tweaked to increase the score.

__Attempt 2:__
```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)
        # dropout
        self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flattening
        x = x.view(-1, 64 * 4 * 4)
        # fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
```
This attempt uses a sequential CNN with more layers and a larger batch size along with a defined flattening layer.
The channels are set to 3 and 16, 16 and 32, and 32 and 64 for the first, second, and third Convolutional Layers, respectively.
In the MaxPool layer, it is changed to downsample the input representation by taking the maximum value over the window defined by pool size for each dimension along the features axis. It also includes a 50% dropout layer to reduce overfitting.

** insert images here

These changes led us to an increase of 72% accuracy.
To increase the accuracy more, the next attempt tweaks the hyperparameters more along with the learning rate.

__Attempt 3:__
```
class CNN(nn.Module):
   

    def __init__(self):
        
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x
```
The revised model above has an increased amount of layers, and added Convolutional blocks that have a kernel size of 3. 
Channel sizes are also adjusted.

This attempt also changes the learning rate from 0.01 to 0.001 so that our model converges more gradually.
```
import torch.optim as optim
# specify loss function
criterion = nn.CrossEntropyLoss()
# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=.001)
```

** insert images

Now the accuracy has improved to 82%. 
The batches in the previous models were too small in proportion to our large dataset.
The higher learning rate from before prevented the model from reaching the minimum loss in 30 epochs. Changing it to 0.001 helps it converge much more quickly.

### Model comparison

** insert images

### Challenges
1. __Feature extraction:__ Observe how the class accuracies consistently have lower accuracies for images of cats, dogs, and birds. This is likely because of these images have more colours, which suggests an issue in feature extraction. 
2. __Too many epochs and lower learning rates:__ When the number of epochs is increased with a lower learning rate, the model overfits and the accuracy drops.
3. __Tuning hyperparameters:__ This is very time-consuming as training itself takes a very long time.
