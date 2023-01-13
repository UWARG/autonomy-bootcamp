import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from load_Data import train_iter

images = train_iter.next()

def show_img(img):
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


plt.figure(figsize=(24,12))
show_img(torchvision.utils.make_grid(images))