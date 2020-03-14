import torch

import torch.nn as nn
import torch.nn.functional as F

# _line_shape = 95 * 8 * 10
_line_shape = 4620


# article: 500 epochs
#  validation stopiing
# first kernel: (1,64) where the kernel should be half the samplign ratre
#
class ConvolNetSpatial(nn.Module):
    def __init__(self):
        super(ConvolNetSpatial, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(1, 64), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(2, 1), stride=1, padding=1)
        self.conv2_drop1 = nn.Dropout2d(p=0.25)
        self.conv2_drop2 = nn.Dropout2d(p=0.25)
        self.conv2_avg_pool = nn.AvgPool2d(2)
        self.conv2_bn1 = nn.BatchNorm2d(20)
        self.conv2_bn2 = nn.BatchNorm2d(10)

        self.fc1 = torch.nn.Linear(_line_shape, 2)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_bn1(x)
        x = F.relu(x, True)
        x = self.conv2_avg_pool(x)
        x = x.view(-1,_line_shape)  # flatten
        x = self.fc1(x)
        # x = self.conv2_drop2(x)

        # x = self.fc2(x)
        return F.log_softmax(x, dim=1)
