from random import random, randint

import torch
from time import time
import matplotlib.pyplot as plt

import torch.nn as nn            # containing various building blocks for your neural networks
import torch.optim as optim      # implementing various optimization algorithms
import torch.nn.functional as F  # a lower level (compared to torch.nn) interface

# torchvision: popular datasets, model architectures, and common image transformations for computer vision.
import torchvision
# transforms: transformations useful for image processing
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

import glob
import os.path as osp
import numpy as np
from PIL import Image


class MNIST(Dataset):
    """
    A customized data loader for MNIST.
    """

    def __init__(self,
                 root,
                 transform=None,
                 preload=False):
        """ Intialize the MNIST dataset

        Args:
            - root: root directory of the dataset
            - tranform: a custom tranform function
            - preload: if preload the dataset into memory
        """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        for i in range(10):
            filenames = glob.glob(osp.join(root, str(i), '*.png'))
            for fn in filenames:
                self.filenames.append((fn, i))  # (filename, label) pair

        # if preload dataset into memory
        if preload:
            self._preload()

        self.len = len(self.filenames)

    def _preload(self):
        """
        Preload dataset to memory
        """
        self.labels = []
        self.images = []
        for image_fn, label in self.filenames:
            # load images
            image = Image.open(image_fn)
            self.images.append(image.copy())
            # avoid too many opened files bug
            image.close()
            self.labels.append(label)

    # probably the most important to customize.
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.images is not None:
            # If dataset is preloaded
            image = self.images[index]
            label = self.labels[index]
        else:
            # If on-demand data loading
            image_fn, label = self.filenames[index]
            image = Image.open(image_fn)

        # May use transform function to transform samples
        # e.g., random crop, whitening
        if self.transform is not None:
            image = self.transform(image)
        # return torch.rand(1, 30,30), randint(0, 9)
        return image , label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
        #        dilation=1, groups=1, bias=True, padding_mode='zeros')
        # in_channels = D, the depth of the input image, 3 for RGB and 1 for greyscale
        # out channles= number of filters
        # kernel size = K, size of the filter
        # Size of the kernel layer: D=1, K=5 ==> 5X5X1 + 1 = 26
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        # Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.max_pool = nn.MaxPool2d(2)
        # ReLU(inplace=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Note: the following two ways for max pooling / relu are equivalent.
        # 1) with torch.nn.functional:
        print("x shape:", x.size())
        conv_ = self.conv1(x)
        # output volume:
        # W = HighxWidth = 28x28 = 784
        # F = K = 5
        # P = 0
        # S = 1
        # ((W−F+2P)/S+1 = (784-5 + 2*0)/1 + 1 = 896
        print("conve shape:", conv_.size())
        max_pool = F.max_pool2d(conv_, 2)
        print("max pool shape:",max_pool.size())
        x = F.relu(max_pool)
        print("rlue shape:",x.size())

        # 2) with torch.nn:
        self_conv_ = self.conv2(x)
        print("conve shape:", self_conv_.size())
        drop = self.conv2_drop(self_conv_)
        print("drop shape:", drop.size())
        x = self.relu(self.max_pool(drop))
        print("x relu :", x.size())
        x = x.view(-1, 320)
        print("x view:", x.size())
        x = F.relu(self.fc1(x))
        print("x rele:", x.size())
        x = F.dropout(x, training=self.training)
        print("x drop 2:", x.size())
        x = self.fc2(x)
        print("x fc:", x.size())
        softmax = F.log_softmax(x, dim=1)
        print("x fc:", softmax.size())
        return softmax



trainset = MNIST(
    root='/Users/shiran.s/dev/p300_net/data/mnist_png/training_small',
    preload=True, transform=transforms.ToTensor(),
)

# Use the torch dataloader to iterate through the dataset
# We want the dataset to be shuffled during training.
trainset_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)

# Load the testset
testset = MNIST(
    root='/Users/shiran.s/dev/p300_net/data/mnist_png/testing_small',
    preload=True, transform=transforms.ToTensor(),
)
# Use the torch dataloader to iterate through the dataset
testset_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=1)

use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")

model = Net().to(device)
params = list(model.parameters())
for tensor in params:
    print('Layer {}: {} elements'.format(tensor.name, torch.numel(tensor)))

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



# functions to show an image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some random training images
dataiter = iter(trainset_loader)
images, labels = dataiter.next()

print("images shape:",images.size())
print(images[0].size())
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % labels[j] for j in range(16)))
images.size()


def train(epoch, log_interval=100):
    model.train()  # set training mode
    iteration = 0
    for ep in range(epoch):
        start = time()
        for batch_idx, (data, target) in enumerate(trainset_loader):
            # bring data to the computing device, e.g. GPU
            data, target = data.to(device), target.to(device)

            print("data shape:", data.shape)
            print("target shape:", target.shape)
            print(model)

            # forward pass
            output = model(data)
            # compute loss: negative log-likelihood
            loss = F.nll_loss(output, target)

            # backward pass
            # clear the gradients of all tensors being optimized.
            optimizer.zero_grad()
            # accumulate (i.e. add) the gradients from this forward pass
            loss.backward()
            # performs a single optimization step (parameter update)
            optimizer.step()

            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                        100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1

        end = time()
        print('{:.2f}s'.format(end - start))
        test()  # evaluate at the end of epoch

def test():
    model.eval()  # set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))

train(5)  # train 5 epochs should get you to about 97% accuracy