import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm

from net import Net

class Model:
    def __init__(self, device, train_set, test_set, train_loader, test_loader):
        self.device = device
        self.train_set = train_set
        self.test_set = test_set
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = 10
        self.eval_losses = []
        self.eval_accuracy = []
        self.train_losses = []
        self.train_accuracy = []
        self.model = None
        self.loss_function = None
        self.optimizer = None

    def build_cnn(self):
        net = Net()
        self.model = net.get_model()
        self.model.to(self.device)

    def set_loss_opt(self):
        if self.model is None:
            self.build_cnn()

        # initialize loss function: measures disparity between obtained result and target value
        self.loss_function = nn.CrossEntropyLoss()
        
        # initialize optimizer: adjusts model parameters and reduces error with each epoch
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def train_model(self):
        if self.loss_function is None or self.optimizer is None:
            self.set_loss_opt()

        for epoch in range(self.epochs):
            print('\nEpoch : %d' % (epoch + 1))
            self.__train()
            self.__test()

    # trains the model
    def __train(self):
        training_loss = 0
        correct_pred = 0
        total_pred = 0
        self.model.train()

        for data in tqdm(self.train_loader):
            inputs = data[0].to(self.device)
            labels = data[1].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)
            
            # Backpropagation: fine-tunes the weights for each layer based on the loss obtained per iteration
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            training_loss += loss.item()

            # the class with the highest energy is what we choose as prediction
            predicted = torch.max(outputs.data, 1)[1]
            correct_pred += (predicted == labels).sum().item()
            total_pred += labels.size(0)
            
        training_loss /= len(self.train_loader)
        self.train_accuracy.append(100 * float(correct_pred) / total_pred)
        self.train_losses.append(training_loss)
        print('Training Loss: %.4f | Accuracy: %.4f' % (training_loss, self.train_accuracy[-1]))

    # evaluates the model's performance against test data
    def __test(self):
        test_loss = 0
        correct_pred = 0
        total_pred = 0
        self.model.eval()

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                images = data[0].to(self.device)
                labels = data[1].to(self.device)
                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)
                
                test_loss += loss.item()

                # the class with the highest energy is what we choose as prediction
                predicted = torch.max(outputs.data, 1)[1]
                correct_pred += (predicted == labels).sum().item()
                total_pred += labels.size(0)

        test_loss /= len(self.test_loader)
        self.eval_losses.append(test_loss)
        self.eval_accuracy.append(100 * float(correct_pred) / total_pred) 
        print('Test Loss: %.4f | Accuracy: %.4f' % (test_loss, self.eval_accuracy[-1]))

    def plot_results(self):
        self.__plot_loss()
        self.__plot_accuracy()

    def __plot_loss(self):
        self.__plot_helper({
            'train' : self.train_losses,
            'eval' : self.eval_losses,
            'title' : 'Loss vs. # of Epoch',
            'x_label' : '# of Epochs',
            'y_label' : 'Loss',
            'x_legend' : 'Train',
            'y_legend' : 'Test',
            'fig' : 'loss.png'
        })

    def __plot_accuracy(self):
        self.__plot_helper({
            'train' : self.train_accuracy,
            'eval' : self.eval_accuracy,
            'title' : 'Accuracy vs. # of Epoch',
            'x_label' : '# of Epochs',
            'y_label' : 'Accuracy',
            'x_legend' : 'Train',
            'y_legend' : 'Valid',
            'fig' : 'accuracy.png'
        })
    
    def __plot_helper(self, plot_data):
        plt.plot(plot_data['train'], '-o')
        plt.plot(plot_data['eval'], '-o')
        plt.title(plot_data['title'])
        plt.xlabel(plot_data['x_label'])
        plt.ylabel(plot_data['y_label'])
        plt.legend([plot_data['x_legend'], plot_data['y_legend']])
        plt.savefig(plot_data['fig'])
        plt.close()
