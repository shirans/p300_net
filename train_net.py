import torch
import torch.nn as nn

import numpy as np
from nets.two_layers_net import TwoLayersNet
from preprocess.helpers import get_device, load_conf, getModel
from preprocess.preprocess import load_data
import torch.optim as optim

from run_fc.train_model import train_model


# parameters
data_dir, n_epochs = load_conf()

print(f'Using PyTorch v{torch.__version__}')

# load the data
train_loader, valid_loader = load_data(data_dir)


# create a complete CNN
device = get_device()
model = getModel(device)
# Get the device

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0005)

valid_loss_min = np.Inf  # track change in validation loss

train_model(model=model, n_epochs=n_epochs, train_loader=train_loader, valid_loader=valid_loader, device=device,
           optimizer=optimizer, criterion=criterion)
