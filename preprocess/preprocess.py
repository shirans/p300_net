import fnmatch
import torch
import os
from mne import find_events
import mne
import numpy as np
import pandas as pd

from scipy.io import loadmat
from copy_helpers import load_muse_csv_as_raw__copy
from preprocess.helpers import _reject, _event_ids
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)


# preprocess - load the data, split to train,test, set lables
# Digitized to 240Mz, we want 600 MS to identify P300 at the half of it so 250 samples are 600 MS


def label_marker(row):
    if row['StimulusCode'] == 0:
        return 0  # no event
    if row['StimulusCode'] != 0 and row['StimulusType'] == 1:
        return 2  # target
    return 1  # non target


def load_files(data_dir):
    if os.path.isfile(data_dir):
        logger.info("loading a single file from path: {}".format(data_dir))
        return load_samples(loadmat(data_dir))
    num_files = len(os.listdir(data_dir))
    logger.info("loading {} files from path: {}".format(num_files, data_dir))
    dfs = []
    for file in os.listdir(data_dir):
        if fnmatch.fnmatch(file, 'A*'):
            mat = loadmat(data_dir + file)
            try:
                df = load_samples(mat)
                dfs.append(df)
                logger.info("added file: {}".format(file))
            except Exception as e:
                logger.info("failed with file {} with error: {}".format(file, str(e)))
    return pd.concat(dfs)


def load_samples(x):
    samplenr = x['samplenr'][:, 0]
    signal = x['signal']
    StimulusType = x['StimulusType']
    StimulusCode = x['StimulusCode']
    trialnr = x['trialnr']

    # choose spcific columns
    # z_based = [42, 24, 29, 44, 45]
    # signal = signal[:, z_based]

    sig = np.column_stack((samplenr, signal, StimulusCode, StimulusType, trialnr))
    df = pd.DataFrame(sig)
    y = ['samplenr'] + ['sig_' + str(a) for a in range(1, 65)] + ['StimulusCode', 'StimulusType', 'trialnr']
    df.columns = y
    return df


def load_data_choose_columns(input_path, output_path):
    # TODO: split to train, validate, test
    df = load_files(input_path)

    df = df.astype(float)
    df['Marker'] = df.apply(lambda row: label_marker(row), axis=1)
    # df = df[indexes_to_include + ['Marker']]
    # use all 64 chammels
    logger.info("saving data to file: {}".format(output_path))
    df.to_csv(path_or_buf=output_path, index=False)


def load_data(input_path, replace_ch_names, ch_ind, reject=_reject, event_ids=_event_ids):
    raw = load_muse_csv_as_raw__copy([input_path], sfreq=240, stim_ind=-1, replace_ch_names=replace_ch_names,
                                     ch_ind=ch_ind)

    raw.filter(1, 30, method='iir')

    events = find_events(raw)

    epochs = mne.Epochs(raw, events=events, event_id=event_ids,
                        tmin=-0.1, tmax=0.8, baseline=None,
                        reject=reject, preload=True,
                        verbose=False)
    print('sample drop %: ', (1 - len(epochs.events) / len(events)) * 100)

    labels = epochs.events[:, -1]

    x = epochs.get_data()  # format is in (trials, channels, samples)
    x = x.astype(np.double)
    y = labels

    valid_size = 0.2  # percentage of training set to use as validation
    num_train = len(x)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # Define samplers for obtaining training and validation batches
    from torch.utils.data.sampler import SubsetRandomSampler

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_data = []
    for i in range(len(x)):
        train_data.append([x[i], y[i]])

    print("X shape: {}".format(x.shape))
    print("Y shape: {}".format(y.shape))

    valid_size = 0.2

    batch_size = 64  # how many samples per batch to load
    num_workers = 0
    num_train = len(x)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)

    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)

    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print(type(images))
    print(images.shape)
    print(labels.shape)

    return train_loader, valid_loader
