# %%
from itertools import cycle

import numpy as np
import torch
from torch import nn
import torch.optim as optim

from matplotlib import pyplot as plt

# I decided I liked convnext better, being a newer architecture
# released with weights
# although model.py still has my comments, I did not bother with
# modelv2.py, since the architectural changes are already listed
# in their paper
from modelv2 import convnext_small
from build_dataset import get_dataset, CLASS_REPR

# parameters for training
epochs       = 3
loss_file    = 'losses.txt'
plot_file    = 'plots.png'

batch_size   = 32
lr           = 1e-5
architecture = 'resnet18'
num_classes  = len(CLASS_REPR)

# I want to fine-tune the model, since this results in higher
# total accuracy and is more energy efficient to train
model = convnext_small(pretrained=True, num_classes=10).cuda()

# building the dataset, see build_dataset for more details
train_loader, test_loader = get_dataset(batch_size, augment=True)
# make the test_loader be an infinite cycle, so StopIteration never occurs
test_loader_t = cycle(iter(test_loader))

# construct the losses, in this case, the negative log-likehood loss
# which is used for classification tasks. It is the log of the softmax
# of the logits multiplied by the label. I don't want to implement label
# smoothing here, since its over-kill
criterion = nn.CrossEntropyLoss()

# Use the adam optimizer for lesser hyper-parameters
optimizer = optim.Adam(model.parameters(), lr=lr)

# save the loss ever log_steps
log_step     = 500
val_losses   = []
train_losses = []
loss_msg     = []
# start the training loop
# in a more complex project, I would usually separate this
# in a separate function or class

total_steps = 0
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    running_val_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # set the model to the train stage, since sometimes
        # dropout has different behaviors
        model.train()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # move data to gpu for acceleration
        inputs = inputs.cuda()
        labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # also compute the evalulation loss
        with torch.no_grad():
            e_x, e_y = next(test_loader_t)
            model.eval()
            eval_loss = criterion(model(e_x.cuda()), e_y.cuda())
            # transfer back to cpu
            running_val_loss += float(eval_loss)

        # print statistics
        running_loss += loss.item()
        if i % log_step == (log_step - 1):    # print every log_step mini-batches
            train_msg = f'[{epoch + 1}, {i + 1:5d}] train loss: {running_loss / log_step:.3f} \n'
            eval_msg = f'[{epoch + 1}, {i + 1:5d}] eval loss: {running_val_loss / log_step:.3f} \n'
            print(train_msg)
            print(eval_msg)

            loss_msg.append(train_msg)
            loss_msg.append(eval_msg)
            train_losses.append(running_loss / log_step)
            val_losses.append(running_val_loss / log_step)

            running_loss = 0.0
            running_val_loss = 0.0
        total_steps += 1


print('Finished Training')

# plot losses over time
plt.plot(np.array(train_losses), label='train-loss')
plt.plot(np.array(val_losses), label='val-loss')
plt.xlabel("steps")
plt.ylabel('mean-crossentropy-loss')
plt.title('mean-crossentropy-loss over steps')
plt.legend()
plt.savefig(plot_file)
plt.show()

# %%
# save losses in a file
with open(loss_file, 'w') as f:
    for t_msg in loss_msg:
        f.write(t_msg)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images.cuda())
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += float((predicted == labels.cuda()).sum().item())


print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# save accuracy in a file
with open('final_accuracy.txt', 'w') as f:
    f.write(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
