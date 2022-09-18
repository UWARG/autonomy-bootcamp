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
from cgi import test
from unittest import result
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
trainLoss = []
valLoss = []
for epoch in range(11):
  print('Epoch ' + str(epoch))
  runningLoss = 0.0
  runningValLoss = 0.0
  for i in range(40000):
    # 40000
    optimizer.zero_grad()
    outputs = network(dataTensor[i])
    loss = criterion(outputs, labelTensor[i])
    loss.backward()
    optimizer.step()
    runningLoss += loss.item()
  runningLoss = runningLoss/40000
  trainLoss.append(runningLoss)
  for i in range(10000):
    # 10000
    valOut = network(dataTensor[i+40000])
    loss = criterion(valOut,labelTensor[i+40000])
    runningValLoss += loss.item()
  runningValLoss = runningValLoss/10000
  valLoss.append(runningValLoss)
print(trainLoss)
print(valLoss)

correct = 0
for i in range(10000):
  result = network(dataTensor[i+40000])
  print(torch.argmax(result).item())
  print(labelTensor.data[i+40000].item())
  if(torch.argmax(result).item() == labelTensor.data[i+40000].item()):
    correct += 1

print('Correct: ' + str(correct) + ' (' + str(correct/100.0) + '%) out of 10000')
plt.plot(trainLoss, label = "Training Loss")
plt.plot(valLoss, label = "Value Loss")
plt.legend()
plt.show()