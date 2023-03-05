#This file used to test the performance of the model trained in main.py

import matplotlib.pyplot as plt
import numpy as np
import torch
from main import testLoader, Net, device, classes

net = Net()

PATH = './models/cifar_net.pth'
net.load_state_dict(torch.load(PATH))

dataIteration = iter(testLoader)
images, labels = next(dataIteration)

correctPrediction = {classname: 0 for classname in classes}
total = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in testLoader:
        images, labels = data

        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)

        _, predictions = torch.max(outputs, 1)

        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correctPrediction[classes[label]] += 1
            #Track prediction results of the model 

            total[classes[label]] += 1

for classname, correctCount in correctPrediction.items():
    accuracy = 100*float(correctCount) / total[classname]
    print(f'Accuracy for class {classname}: {accuracy}')
    #Display model accuracy for model


