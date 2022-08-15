import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# nn architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layers
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)
        # dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten
        x = x.view(-1, 64 * 4 * 4)
        # linear layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x) # output so no dropout or relu
        return x