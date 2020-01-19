import torch
from torch import nn




class FC_Network_two_layers(nn.Module):
    def __init__(self):
        super().__init__()

        # define the layers
        self.layers = nn.Sequential(
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        # forward pass
        x = torch.sigmoid(self.layers(x))
        return x


# instantiate the model
model = FC_Network_two_layers()

# print model architecture
print(model)