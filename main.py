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

from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pandas as pd
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt

# Your working code here

# Unpack data
def unpickle(file):
  import pickle
  with open(file, 'rb') as fo:
      dict = pickle.load(fo, encoding='bytes')
  return dict

# Put data into dataframe
dicts = [];
for i in range(1,6):
  dicts.append(unpickle("data_batch_"+str(i)))
dataFrame = pd.DataFrame(columns=['labels','data'])
for i in range(5):
  temp = pd.DataFrame([dicts[i][b'labels'],dicts[i][b'data']])
  temp = temp.transpose()
  temp.columns=['labels','data']
  dataFrame = pd.concat([dataFrame,temp], ignore_index = True)

# Change data into tensor form
labelTensor = torch.tensor(dataFrame['labels'].values, dtype = torch.uint8) 
dataTensor = torch.tensor(dataFrame['data'].values, dtype = torch.uint8)
dataTensor = torch.div(dataTensor,255)

# Construct neural network
class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.flatten = nn.Flatten()
    self.linearReLUStack = nn.Sequential(
      nn.Linear(3*32*32,512),
      nn.ReLU(),
      nn.Linear(512,512),
      nn.ReLU(),
      nn.Linear(512,10),
    )
  def forward(self,x):
    logits = self.linearReLUStack(x)
    return logits
network = NeuralNetwork()
loss_fn = torch.nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr = 0.001, momentum = 0.9)

# Train

for epoch in range(2):
  plt.subplot(131+epoch)
  trainLoss = []
  valLoss = []
  runningLoss = 0.0
  runningValLoss = 0.0
  for i in range(39999):
    optimizer.zero_grad()
    outputs = network(dataTensor[i])
    loss = criterion(outputs, labelTensor[i])
    loss.backward()
    optimizer.step()
    runningLoss += loss.item()
    if i % 2000 == 1999 or i == 39999:
      print(f'[{epoch + 1}, {i + 1:5d}] loss: {runningLoss / 2000:.3f}')
      trainLoss.append(runningLoss / 2000)
      runningLoss = 0.0
  plt.plot(trainLoss, label = "Training Loss")
  for i in range(9999):
    valOut = network(dataTensor[i+40000])
    loss = criterion(valOut,labelTensor[i+40000])
    runningValLoss += loss.item()
    if i % 2000 == 1999 or i == 49999:
      valLoss.append(runningValLoss / 2000)
      runningValLoss = 0
  plt.plot(valLoss, label = "Value Loss")

plt.legend()
plt.show()