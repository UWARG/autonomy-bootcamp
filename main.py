import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from multiprocessing import freeze_support


if __name__ == '__main__':

	# Makes sure that program can create multiple processes correct
	freeze_support()

	################################################################################
	#   Data Preparation
	#       1. Download the CIFAR-10 dataset
	#       2. Apply transformations such as cropping, flipping, normalization
	#       3. Create dataloaders for training and testing
	################################################################################

	#  transforms.Compose([...]) applies the following transformations:
	#  - The image is randomly flipped horizontally
	#  - The image is converted to a PyTorch tensor
	#  - The image pixel values are normalized to the range [-1, 1] for each input channel
	transform = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])

	# datasets.CIFAR10(...) downloads the CIFAR-10 dataset and applies the transformations defined above
	#  - A dataset 'trainset' is created for training
	#  - A dataset 'testset' is created for testing
	trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

	# torch.utils.data.DataLoader(...) creates a dataloader for the training and testing dataset
	#  - The dataloader 'trainloader' loads the training dataset in batches of 128 images
	#  - The dataloader 'testloader' loads the testing dataset in batches of 128 images
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
	testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

	
	################################################################################
	#   Model Definition
	#       This program defines a simple CNN model for image classification.
	#       The model consists of the following layers:
	#           - Convolutional layer with 16 filters, kernel size 3, padding 1
	#           - Batch normalization layer, which normalizes the output of the convolutional layer
	#           - ReLU activation layer, which applies the ReLU function
	#               - If x < 0, then ReLU(x) = 0; otherwise ReLU(x) = x
	#           - Convolutional layer with 32 filters, kernel size 3, padding 1
	#           - Batch normalization layer
	#           - ReLU activation layer
	#           - Max pooling layer, which downsamples the input by taking the maximum value in each 2x2 window
	#               - By reducing the dimensions of the input, the number of parameters in the model is reduced
	#               - That way, only the most important features are kept
	#           - Fully connected layer of size 32 x 16 x 16 with 10 output units
	#               - Maps the output of the convolutional layer to the 10 classes
	#       The forward pass of the model is defined in the forward(...) function
	#           - The input x is passed through the layers of the model
	#           - x.view() reshapes the output of the convolutional layer to a vector
	#           - The output of the model is returned
	################################################################################

	class SimpleCNN(nn.Module):
		def __init__(self):
			super(SimpleCNN, self).__init__()
			self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
			self.bn1 = nn.BatchNorm2d(16)
			self.relu1 = nn.ReLU(inplace=True)
			
			self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
			self.bn2 = nn.BatchNorm2d(32)
			self.relu2 = nn.ReLU(inplace=True)
			
			self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
			
			self.fc = nn.Linear(32 * 16 * 16, 10)  

		def forward(self, x):
			batch_size = x.size(0)
			x = self.conv1(x)
			x = self.bn1(x)
			x = self.relu1(x)
			
			x = self.conv2(x)  
			x = self.bn2(x)
			x = self.relu2(x)
			
			x = self.pool(x)
			
			x = x.view(batch_size, -1)

			x = self.fc(x)
			return x

	################################################################################
	#   Training
	#       The training process consists of the following steps:
	#           1. Setting the device for computation
	#           2. Creating a model, loss function, and optimizer
	#           3. Configuring the number of epochs and initalizing variables for tracking the training progress
	#           4. Training the model for the specified number of epochs
	################################################################################

	# Setting the device for computation
	#   - If a GPU is available, the device is set to GPU
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Creating a model, loss function, and optimizer
	#   - The loss function is defined as the cross entropy loss
	#       - CrossEntropyLoss(p, q) = - ∑(q_i * log(p_i))
	#   - The optimizer is defined as stochastic gradient descent (SGD)
	#       - SGD updates the parameters of the model on each mini-batch
	#          - Introducing randomness into the training process helps the model avoid local minima
	#       - The learning rate is set to 0.01
	#          - This refers to how quickly the model learns
	#       - The momentum is set to 0.9
	#          - This accelerates the learning process to help the model converge faster
	model = SimpleCNN().to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

	num_epochs = 30
	train_losses = []

	for epoch in range(num_epochs):
		model.train()
		train_loss = 0.0

		for batch_idx, (data, target) in enumerate(trainloader):
			data, target = data.to(device), target.to(device)
			
			# Resets the gradients to 0
			optimizer.zero_grad()
			
			output = model(data)

			# Computes the loss
			loss = criterion(output, target)
			loss.backward()

			# Updates the parameters of the model
			optimizer.step()

			# Accumulates the loss for the epoch
			train_loss += loss.item()

			# Prints progress information
			if batch_idx % 12 == 0:
				print('Epoch: {} [{}/{} ({:.0f}%)]'.format(
					epoch + 1, batch_idx * len(data), len(trainloader.dataset),
					100. * batch_idx / len(trainloader)))

		
		# Prints the average loss for the epoch 
		train_loss /= len(trainloader.dataset)
		train_losses.append(train_loss)
		print('Epoch: {} Train loss: {:.6f}'.format(epoch + 1, train_loss))

	################################################################################
	#   Evaluation and Plotting
	#       The evaluation process consists of the following steps:
	#           1. Setting the model to evaluation mode
	#           2. Disabling gradient computation
	#           3. Computing the average loss and accuracy on the test set
	#           4. Plotting the training loss over time
	################################################################################

	model.eval()

	test_loss = 0.0
	correct = 0

	# Disables gradient computation
	with torch.no_grad():
		for data, target in testloader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += criterion(output, target).item()

			_, predicted = torch.max(output, 1)
			correct += (predicted == target).sum().item()

	test_loss /= len(testloader.dataset)
	accuracy = 100.0 * correct / len(testloader.dataset)

	# Prints the average loss and accuracy on the test set
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(testloader.dataset), accuracy))
	

	# Plot the training loss over time
	plt.plot(train_losses, label='Training Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()
