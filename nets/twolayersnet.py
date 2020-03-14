import torch.nn as nn
import torch.nn.functional as F

_line_shape = 1*6*217


# define the CNN architecture
class TwoLayersNet(nn.Module):
    def __init__(self):
        super(TwoLayersNet, self).__init__()
        self.fc1 = nn.Linear(_line_shape, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = x.view(-1, _line_shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
