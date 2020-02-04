import torch.nn as nn

import numpy as np
from preprocess.helpers import get_device, getModel, load_conf_train
import torch.optim as optim

from preprocess.preprocess import data_to_raw
from run_fc.train_model import train_model

# parameters
data_dir, n_epochs, model_name = load_conf_train()

# load the data
train_loader, valid_loader = data_to_raw(data_dir)

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
            optimizer=optimizer, criterion=criterion, model_name=model_name)
