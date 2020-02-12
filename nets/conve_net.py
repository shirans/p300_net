import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=2)
        self.conv_3 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=2)

    def forward(self, x):
        # define forward pass
        shap = x.shape
        x = x.reshape((shap[0], 1, shap[1], shap[2]))
        x = F.relu(self.conv_1(x))
        size = x.size()
        print("size:", size)
        x = F.relu(self.conv_2(x))
        size = x.size()
        print("size:", size)
        x = F.relu(self.conv_3(x))
        size = x.size()
        print("size:", size)
        return x
