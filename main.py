import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import ssl
import random
import os
import time

from models import *
import config

ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: %s" % (device))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(2021)

#######################################################################
print("preparing data...")

best_acc = 0
start_epoch = 0

# If running on Windows and you get a BrokenPipeError, try setting
# the num_worker of torch.utils.data.DataLoader() to 0.
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=config.transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=config.train_batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=config.transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=config.test_batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

#######################################################################
print("defining a convolutional neural network...")
net = config.net
net.to(device)
if config.resume:
    # Load checkpoint.
    print('Resuming from checkpoint...')
    assert os.path.isdir(
        config.persist_dir_name), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./%s/%s_ckpt.pth' %
                            (config.persist_dir_name, net.__class__.__name__))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']+1
#######################################################################
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=5e-4)
# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

losses = []
losses_epoch = []
accuracy_epoch = []


def train(epoch):
    print('\nEpoch: %d' % (epoch+1))
    net.train()
    running_loss = 0.0
    running_loss_total = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_loss_total += loss.item()
        # print every train_show_interval mini-batches
        if i % config.train_show_interval == config.train_show_interval-1:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / config.train_show_interval))
            losses.append(running_loss / config.train_show_interval)
            running_loss = 0.0
    losses_epoch.append(running_loss_total/len(trainloader))


def test(epoch):
    global best_acc
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            #images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print('[Epoch %d] Accuracy of the network on the 10000 test images: %d %%' % (epoch+1,
                                                                                  acc))
    accuracy_epoch.append(acc)
    if config.save_flag and acc > best_acc:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(config.persist_dir_name):
            os.mkdir(config.persist_dir_name)
        torch.save(state, './%s/%s_ckpt.pth' %
                   (config.persist_dir_name, net.__class__.__name__))
        best_acc = acc


print("train and test...")
start_time = time.time()
for epoch in range(start_epoch, start_epoch + config.epoch_num):
    train(epoch)
    test(epoch)
    scheduler.step()
end_time = time.time()
total_time = end_time - start_time
print("total time of %d epoches: %fs\naverage per epoch: %fs" %
      (config.epoch_num, total_time, total_time/config.epoch_num))

#######################################################################
print("showing statistics...")
if not os.path.isdir(config.img_path):
    os.mkdir(config.img_path)

x = range(0, len(losses))
plt.plot(x, losses)
plt.title("batch size: %d, learning rate:%f, epoch: %d to %d" %
          (config.train_batch_size, config.learning_rate, start_epoch+1, start_epoch+config.epoch_num))
plt.xlabel("per %d batches" % (config.train_show_interval))
plt.ylabel("loss")
plt.savefig("%sloss_%s_epoch%dto%d" % (config.img_path,
            net.__class__.__name__, start_epoch+1, start_epoch+config.epoch_num))
plt.show()

x = range(start_epoch+1, len(losses_epoch)+1)
plt.plot(x, losses_epoch)
plt.title("batch size: %d, learning rate:%f loss of train set" %
          (config.train_batch_size, config.learning_rate))
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("%sloss_%s" % (config.img_path,
            net.__class__.__name__))
plt.show()

x = range(start_epoch+1, len(accuracy_epoch)+1)
plt.plot(x, accuracy_epoch)
plt.title("batch size: %d, learning rate:%f acc of test set" %
          (config.test_batch_size, config.learning_rate))
plt.xlabel("epoch")
plt.ylabel("accuracy/%")
plt.savefig("%sacc_%s" % (config.img_path,
            net.__class__.__name__))
plt.show()
