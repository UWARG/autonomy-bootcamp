from typing import Dict, List

import matplotlib.pyplot as plt

from typing import Dict, List
import matplotlib.pyplot as plt

def plot_loss_curves(results: Dict[str, List[float]]):
    loss = results['train_loss']
    test_loss = results['test_loss']

    acc = results['train_acc']
    test_acc = results['test_acc']

    number_of_epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15, 7))

    # Plot loss curve
    plt.subplot(1, 2, 1)
    plt.plot(number_of_epochs, loss, label='train_loss')
    plt.plot(number_of_epochs, test_loss, label='test_loss')
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    
    # Plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(number_of_epochs, acc, label='train_accuracy')
    plt.plot(number_of_epochs, test_acc, label='test_accuracy')
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()
