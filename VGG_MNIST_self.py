import torch
import torchvision
import torchvision.transforms as tf
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np


EPOCH = 1
batch_size_train = 4
batch_size_test = 10
learning_rate = 0.01
momentum = 0.5
log_interval = 1000
# 1. prepare dataset


def getData():
    transforms = tf.Compose([tf.Resize(224), tf.ToTensor(), tf.Normalize((0.5,), (0.5,))])
    # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of
    # shape (C x H x W) in the range [0.0, 1.0].

    trainset = torchvision.datasets.MNIST('./files/', train=True, download=True, transform=transforms)
    # Resize(int or sequence(h*w), interpolation=2). If original img h>w, arg is int,
    # then smaller edge = int -->(int*h/w, int)
    # ToTensor --> Convert a PIL Image or numpy.ndarray to tensor.
    # Normalize(mean, std) --> every channel, input[channel] = (input[channel] - mean[channel]) / std[channel]

    testset = torchvision.datasets.MNIST('./files/', train=False, download=True, transform=transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=True)
    return train_loader, test_loader

    # examples = enumerate(test_loader)
    # batch_idx, (example_data, example_targets) = next(examples)
    # print(len(example_data), example_data.shape)
    # # example_data contains images. example_targets contains images' labels (1,2,3,cat)

# 2. build network

# network
def vgg_block(num_convs, in_channels, num_channels):
    layers = []
    for i in range(num_convs):
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=3, padding=1)]
        in_channels = num_channels
    layers += [nn.ReLU()]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
        layers = []
        for (num_convs, in_channels, num_channels) in self.conv_arch:
            layers += [vgg_block(num_convs, in_channels, num_channels)]
        self.features = nn.Sequential(*layers)
        self.dense1 = nn.Linear(512 * 7 * 7, 4096)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 7 * 7) # make it the same shape as output
        x = self.dense3(self.drop2(F.relu(self.dense2(self.drop1(F.relu(self.dense1(x)))))))
        return x


net = VGG().cuda()
# print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

train_loader, test_loader = getData()

# # show image
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.cpu().numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(EPOCH + 1)]


def train(epoch):
    time_s = time.time()
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()  # len(data)=4, len(train_loader.dataset)=60000
        target = target.cuda()
        optimizer.zero_grad()
        output = net(data)
        # print(output)
        # print(target)
        # break
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()  # propagate gradients collected by backward() back into each network
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * batch_size_train) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(net.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')
    time_e = time.time()
    print('Train time in EPOCH %d is %ds' % (epoch, time_e - time_s))


def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = net(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    # test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


test()
for epoch in range(1, EPOCH + 1):
    train(epoch)
    test()




