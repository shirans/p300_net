import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict
from random import random, randint
import torch.nn as nn

from mne import Epochs, find_events
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from scipy.io import loadmat
from torch.utils.data import SubsetRandomSampler

from copy_helpers import load_muse_csv_as_raw__copy

print(f'Using PyTorch v{torch.__version__}')

# preprocess - load the data, split to train,test, set lables
# Digitized to 240Mz, we want 600 MS to identify P300 at the half of it so 250 samples are 600 MS


# TODO: split to train, validate, test
# TODO: load all data
times = 250
channels = 64
x = loadmat('/Users/shiran.s/dev/p300_net/data/IIb/AAS010R02.mat')

samplenr = x['samplenr'][:, 0]
signal = x['signal']
StimulusType = x['StimulusType']
StimulusCode = x['StimulusCode']
trialnr = x['trialnr']

# choose spcific columns
z_based = [42, 24, 29, 44, 45]
signal = signal[:, z_based]

sig = np.column_stack((samplenr, signal, StimulusCode, StimulusType, trialnr))
df = pd.DataFrame(sig)


def label_marker(row):
    if row[6] == 0:
        return 0  # no event
    if row[6] > 0 and row[7] == 1:
        return 2  # target
    return 1  # non target


df = df.astype(float)
df['Marker'] = df.apply(lambda row: label_marker(row), axis=1)

indexes_to_include = list(range(0, 7))
df = df[indexes_to_include + ['Marker']]
replace_ch_names = None

# use all 64 chammels
replace_ch_names = {'1': "TP9"}
ch_ind = indexes_to_include[:-1]

temp_path = "/Users/shiran.s/dev/p300_net/output/temp_csv.csv"
df.to_csv(path_or_buf=temp_path, index=False)
conditions = OrderedDict()
conditions['Non-target'] = [1]
conditions['Target'] = [2]
event_ids = {'Non-Target': 1, 'Target': 2}

raw = load_muse_csv_as_raw__copy([temp_path], sfreq=240, stim_ind=-1, replace_ch_names=replace_ch_names, ch_ind=ch_ind)

raw.filter(1, 30, method='iir')

events = find_events(raw)

reject = {'eeg': 100e-4}
event_ids = {'Non-Target': 1, 'Target': 2}

epochs = mne.Epochs(raw, events=events, event_id=event_ids,
                    tmin=-0.1, tmax=0.8, baseline=None,
                    reject=reject, preload=True,
                    verbose=False, picks=ch_ind)
print('sample drop %: ', (1 - len(epochs.events) / len(events)) * 100)

labels = epochs.events[:, -1]

valid_size = 0.2  # percentage of training set to use as validation

X = epochs.get_data()  # format is in (trials, channels, samples)
X = X.astype(np.double)
y = labels

num_train = len(X)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# Define samplers for obtaining training and validation batches
from torch.utils.data.sampler import SubsetRandomSampler

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_data = []
for i in range(len(X)):
    train_data.append([X[i], y[i]])

print("X shape: {}".format(X.shape))
print("Y shape: {}".format(y.shape))

# train

import torch.optim as optim

import torch.nn.functional as F


# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Some helpful values
        self.fc1 = nn.Linear(217, 6)
        self.fc2 = nn.Linear(6, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        # define forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# create a complete CNN
model = Net()
model = model.double()
# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0005)

n_epochs = 5000  # you may increase this number to train a final model

valid_loss_min = np.Inf  # track change in validation loss

# Get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
model.to(device)
tLoss, vLoss = [], []
valid_size = 0.2

batch_size = 64  # how many samples per batch to load
num_workers = 0
num_train = len(X)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           sampler=valid_sampler, num_workers=num_workers)

dataiter = iter(train_loader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

for epoch in range(n_epochs):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    #########
    # train #
    #########
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        data = data.to(device)
        target = target.to(device)

        # TODO REMOVE?
        shape = target.shape[0]
        target = target.reshape(shape, 1)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item() * data.size(0)

    # test + graphs
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available

        shape = target.shape[0]
        target = target.reshape(shape, 1)

        data = data.to(device)
        target = target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss
        valid_loss += loss.item() * data.size(0)

    # calculate average losses
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)
    tLoss.append(train_loss)
    vLoss.append(valid_loss)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss

    # Plot the resulting loss over time
plt.plot(tLoss, label='Training Loss')
plt.plot(vLoss, label='Validation Loss')
plt.legend()
plt.show()

