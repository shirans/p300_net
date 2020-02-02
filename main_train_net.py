import os
import torch

import torch.nn as nn

from preprocess.helpers import get_device, load_conf_train, build_model, first_4_columns, first_4_col_names, muse_cols, \
    muse_cols_names
from preprocess.preprocess import load_data
import torch.optim as optim

from run_fc.eval_model import evaluate
from run_fc.train_model import train_model, build_model_path

# parameters
input_path, n_epochs, model_name = load_conf_train()

# load the data

train_loader, valid_loader = load_data(input_path, replace_ch_names=first_4_col_names, ch_ind=first_4_columns)
# train_loader, valid_loader = load_data(input_path, replace_ch_names=muse_cols_names, ch_ind=muse_cols)

# create a complete CNN
device = get_device()
model = build_model(device)
# Get the device

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0005)

train_model(model=model, n_epochs=n_epochs, train_loader=train_loader, valid_loader=valid_loader, device=device,
            optimizer=optimizer, criterion=criterion, model_name=model_name)

model_dir = build_model_path(model_name)
print('loading model from path: {}'.format(model_dir))
model = build_model(device)
model.load_state_dict(torch.load(model_dir))
model.eval()

evaluate(train_loader, device, model)
evaluate(valid_loader, device, model)
