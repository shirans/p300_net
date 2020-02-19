import torch

import torch.nn as nn
import torch.nn.functional as F

_line_shape = 270
# define the CNN architecture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = torch.nn.Linear(_line_shape, 50)
        self.fc2 = nn.Linear(50, 10)

        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # define forward pass
        # print("x shape:", x.size()) # [64, 1, 6, 217]
        conv1 = self.conv1(x)
        # print("conv shape:", conv1.size()) # 64, 10, 6, 217]
        x = F.relu(F.max_pool2d(conv1, 2))
        # print("after max pool relue shape:", x.size()) # [64, 10, 3, 108]
        x = self.relu(self.max_pool(self.conv2_drop(self.conv2(x))))
        # print("after max pool relue shape:", x.size()) # [64, 20, 1, 54]
        # therefor _lineshape = 20*1*54=1080
        x = x.view(-1, _line_shape)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
