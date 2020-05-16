import torch.nn as nn
import torch.nn.functional as F

_line_shape = 1*6*217


# define the CNN architecture
class TwoLayersNet(nn.Module):
    def __init__(self, num_classes):
        super(TwoLayersNet, self).__init__()
        self.fc1 =None
        self.fc2 = nn.Linear(10, num_classes)
        self.reshape_before_fc=None

    def forward(self, x):
        if self.reshape_before_fc is None:
            self.reshape_before_fc=x.shape[1]*x.shape[2]*x.shape[3]
            self.fc1 = nn.Linear(self.reshape_before_fc, 10)
        x = x.view(-1, self.reshape_before_fc)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
