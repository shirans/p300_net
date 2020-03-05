import time
import torch.nn as nn

import numpy as np
from torchsummary import summary

from preprocess.helpers import get_device, get_model, load_conf_train, conf
import torch.optim as optim

from preprocess.preprocess import build_dataloader, train_shapes
from run_fc.eval_model import evaluate
from run_fc.metadata import Metadata
from run_fc.train_model import train_model, train_svm, build_model_path

# parameters
data_dir, n_epochs, model_name, network, stop_condition = load_conf_train()

start = time.time()
# train_svm(data_dir)

output_path, info_path = build_model_path(model_name)
metadata = Metadata(info_path)
metadata.append("config:")
metadata.append(str(conf()))
# load the data
train_loader, valid_loader = build_dataloader(data_dir, 64, metadata,False)

# train_shapes(train_loader)
# create a complete CNN
device = get_device()
model = get_model(device, network)
# Get the device

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.00005)

valid_loss_min = np.Inf  # track change in validation loss
metadata.append(str(model))
train_model(model=model, n_epochs=n_epochs, train_loader=train_loader, valid_loader=valid_loader, device=device,
            optimizer=optimizer, criterion=criterion, stop_condition=stop_condition, output_path=output_path, metadata=metadata)

evaluate(train_loader, device, model, 'train', metadata)
evaluate(valid_loader, device, model, 'valid', metadata)
end = time.time()
# summary(model.type(torch.FloatTensor), (1, 6, 217))
seconds = end - start
metadata.close()
metadata.append("Done! took {} minutes".format(seconds / 60.0))
