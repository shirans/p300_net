import time
import torch.nn as nn

import numpy as np
from preprocess.helpers import get_device, getModel, load_conf_train
import torch.optim as optim

from preprocess.preprocess import build_dataloader, train_shapes
from run_fc.eval_model import evaluate
from run_fc.train_model import train_model, build_folder_path

# parameters
data_dir, n_epochs, model_name = load_conf_train()

# load the data
train_loader, valid_loader = build_dataloader(data_dir)

train_shapes(train_loader)
# create a complete CNN
device = get_device()
model = getModel(device)
# Get the device

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0005)

valid_loss_min = np.Inf  # track change in validation loss
start = time.time()

train_model(model=model, n_epochs=n_epochs, train_loader=train_loader, valid_loader=valid_loader, device=device,
            optimizer=optimizer, criterion=criterion, model_name=model_name)

evaluate(train_loader, device, model, 'train')
evaluate(valid_loader, device, model, 'valid')
end = time.time()
seconds = end - start
print("Done! took {} minutes".format(seconds / 60.0))
