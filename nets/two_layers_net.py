import torch.nn as nn
import torch.nn.functional as F
import torch


# define the CNN architecture
class TwoLayersNet(nn.Module):
    def __init__(self):
        super(TwoLayersNet, self).__init__()

        # Some helpful values
        self.fc1 = nn.Linear(217, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        # define forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x
