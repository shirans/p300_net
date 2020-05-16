import torch

import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=2, stride=1, padding=1)

        self.reshape_before_fc = 10*3*108
        self.fc1 = nn.Linear(self.reshape_before_fc, 10)
        self.fc2 = nn.Linear(10, num_classes)

        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def init_last_later(self, reshape_before_fc):
        self.reshape_before_fc = reshape_before_fc
        self.fc1 = nn.Linear(self.reshape_before_fc, 10)

    def forward(self, x):
        # define forward pass
        x = F.relu(self.conv1(x), True)
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        # if self.reshape_before_fc is None:
        #     self.reshape_before_fc = x.shape[1] * x.shape[2] * x.shape[3]
        #     self.fc1 = nn.Linear(self.reshape_before_fc, 10)

        x = x.view(-1, self.reshape_before_fc)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
