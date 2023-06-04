<h1 align="center">
  <br>
  <i>CNN Image Classification</i>
  <br>
</h1>

<h4 align="center">Created for the WARG Autonomy Bootcamp.
</h4>

###### This project implements a Convolutional Neural Network (CNN) for image classification. The dataset used was the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The test set's accuracy was 7074/10000 (71%) after 30 epochs.

<p align="center">
  <a href="#model">Model</a> •
  <a href="#training-process">Training Process</a>
</p>

## Model
The model architectured used is a simple CNN architecture with the following layers:

* Convolutional layer: 16 filters, kernel size 3, padding 1
* Batch normalization layer
* ReLU activation layer
* Convolutional layer with 32 filters, kernel size 3, padding 1
* Batch normalization layer
* ReLU activation layer
* Max pooling layer: kernel size 2x2, stride 2
* Fully connected layer: input size 16x16x16, output size 10 (number of classes)

## Training Process

* Data preprocessing: The images are normalized and transformed using data augmentation techniques such as random cropping and horizontal flipping
* Optimization: Stochastic Gradient Descent (SGD) optimizer is used with a learning rate of 0.005 and momentum of 0.9 to minimize the Cross Entropy Loss and accelerate learning

---

> GitHub [@cindehaa](https://github.com/cindehaa) &nbsp;&middot;&nbsp;
> LinkedIn [@cindehaa](https://www.linkedin.com/in/cindehaa/)

