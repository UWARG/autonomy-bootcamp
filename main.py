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

from ast import arg
import cv2
import matplotlib
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from re import L

from zmq import device


# Your working code here
REBUILD_DATA = False
device = torch.device("cuda:0")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = torch.Tensor(x_train).reshape(-1, 32, 32)
y_train = torch.Tensor(y_train)
x_test = torch.Tensor(x_test).reshape(-1, 32, 32)
y_test = torch.Tensor(y_test)
x_train = x_train.view(-1, 32, 32)
x_test = x_test.view(-1, 32, 32) #might need to add operation


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        x = torch.randn(32, 32).view(-1, 1, 32, 32)
        self._to_linear = None
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 10)
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim = 1)


net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr = 0.001)
loss_function = nn.MSELoss()
MODEL_NAME = f"model-{int(time.time())}"

BATCH_SIZE = 100
EPOCHS = 3 

def train(net): 
    with open("model.log", "a") as f: 
        for i in tqdm(range(0, len(x_train), BATCH_SIZE)): 
            x_batch = x_train[i:i+BATCH_SIZE].view(-1, 1, 32, 32)
            y_batch = y_train[i:i+BATCH_SIZE]
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            acc, loss = fwd_pass(x_batch, y_batch, train = True)

def batch_test(net): 
    correct = 0 
    total = 0
    with torch.no_grad():
        x_batch = x_test[:BATCH_SIZE].view(-1, 1, 32, 32)
        y_batch = y_test[:BATCH_SIZE]
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        net.zero_grad()
        outputs = net(x_batch)
        matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y_batch)]
        acc = matches.count(True)/len(matches)
        print("Test Accuracy: ", round(acc, 3))

def fwd_pass(X, y, train = False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)
    if train: 
        loss.backward()
        optimizer.step()
    return acc, loss

train(net)
batch_test(net)

style.use("ggplot")
model_name = MODEL_NAME

def create_acc_loss_graph(model_name):
    contents = open("model.log", "r").read().split("/n")
    times = []
    accuracies = []
    losses = []
    test_accs = []
    test_losses = []
    for c in contents: 
        if model_name in c:
            name, timestamp, acc, loss, test_acc, test_loss = c.split(",")
            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))
            test_accs.append(float(test_acc))
            test_losses.append(float(test_loss))
    fig = plt.figure()
    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (0,0), sharex = ax1)
    ax1.plot(times, accuracies, label = "acc")
    ax1.plot(times, test_accs, label = "test_acc")
    ax1.legend(loc = 2)
    ax2.plot(times, losses, label = "loss")
    ax2.plot(times, test_losses, label = "test_loss")
    plt.show() 

create_acc_loss_graph(model_name)


        

