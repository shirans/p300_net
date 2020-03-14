import torch

import torch.nn as nn
import torch.nn.functional as F

_line_shape =217*6*10

# define the CNN architecture
class ConvolNet(nn.Module):
    def __init__(self):
        super(ConvolNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = torch.nn.Linear(_line_shape, 10)
        self.fc2 = nn.Linear(10, 2)

        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # define forward pass
        x = F.relu(self.conv1(x), 2)
        # print("after max pool relue shape:", x.size()) # [64, 10, 3, 108]
        x = self.relu(self.conv2_drop(self.conv2(x)))
        x = x.view(-1, _line_shape)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


