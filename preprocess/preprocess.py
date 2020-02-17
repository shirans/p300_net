import fnmatch
import torch
import os
from mne import find_events
import mne
import numpy as np
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from scipy.io import loadmat
from copy_helpers import load_muse_csv_as_raw__copy
import logging

from preprocess.helpers import replace_ch_names, _reject, _event_ids

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)


# preprocess - load the data, split to train,test, set lables
# Digitized to 240Mz, we want 600 MS to identify P300 at the half of it so 250 samples are 600 MS


# def label_marker(row):
#     if row[6] == 0:
#         return 0  # no event
#     if row[6] > 0 and row[7] == 1:
#         return 2  # target
#     return 1  # non target


def label_marker(row):
    if row['StimulusCode'] == 0:
        return 0  # no event
    if row['StimulusCode'] != 0 and row['StimulusType'] == 1:
        return 2  # target
    return 1  # non target


def files_to_df(data_dir, indexes):
    if os.path.isfile(data_dir):
        logger.info("loading a single file from path: {}".format(data_dir))
        matrix = loadmat(data_dir)
        return data_dict_to_df(matrix, indexes)
    num_files = len(os.listdir(data_dir))
    logger.info("loading {} files from path: {}".format(num_files, data_dir))
    dfs = []
    num_valid_files = 0
    for file in os.listdir(data_dir):
        if fnmatch.fnmatch(file, 'AAS010*') or fnmatch.fnmatch(file, 'AAS011*'):
            matrix = loadmat(os.path.join(data_dir, file))
            try:
                df = data_dict_to_df(matrix, indexes)
                dfs.append(df)
                logger.info("added file: {}".format(file))
                num_valid_files = num_valid_files + 1
            except Exception as e:
                logger.info("failed with file {} with error: {}".format(file, str(e)))
    return dfs


def data_dict_to_df(x, indexes):
    samplenr = x['samplenr'][:, 0]
    signal = x['signal']
    StimulusType = x['StimulusType']
    StimulusCode = x['StimulusCode']
    trialnr = x['trialnr']

    # choose spcific columns
    # z_based = [42, 24, 29, 44, 45]
    signal = signal[:, indexes]

    sig = np.column_stack((samplenr, signal, StimulusCode, StimulusType, trialnr))
    y = ['samplenr'] + indexes + ['StimulusCode', 'StimulusType', 'trialnr']
    df = pd.DataFrame(sig)
    df.columns = y
    return df


def choose_columns_save_csv(input_path, output_path, indexes):
    logger.info("loading data from path: {}".format(input_path))
    unite_files_and_save('train', input_path, output_path, indexes)
    unite_files_and_save('valid', input_path, output_path, indexes)
    unite_files_and_save('test', input_path, output_path, indexes)


def unite_files_and_save(type, path, output_path, indexes):
    full_input_path = os.path.join(path, type)
    dfs = files_to_df(full_input_path, indexes)
    logger.info("used {} files for {}".format(len(dfs), type))
    df = pd.concat(dfs)
    df = add_marker_choose_columns(indexes, df)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    full_output_path = os.path.join(output_path, type + '.csv')
    print("output columns: {}".format(df.columns))
    logger.info("saving {} data to path:{}".format(type, full_output_path))
    df.to_csv(path_or_buf=full_output_path, index=False)


def load_folder(type, path):
    full_path = os.path.join(path, type + '.csv')
    num_columns_in_file = len(pd.read_csv(full_path).columns)

    # all columns are eeg except the last one which is the marker
    ch_ind = list(range(0, num_columns_in_file))[:-1]
    raw = load_muse_csv_as_raw__copy([full_path], sfreq=240, stim_ind=-1, replace_ch_names=replace_ch_names,
                                     ch_ind=ch_ind)

    raw.filter(1, 30, method='iir')
    events = find_events(raw)

    epochs = mne.Epochs(raw, events=events, event_id=_event_ids,
                        tmin=-0.1, tmax=0.8, baseline=None,
                        reject=_reject, preload=True,
                        verbose=False, picks=ch_ind)
    print('sample drop %: ', (1 - len(epochs.events) / len(events)) * 100)
    labels = epochs.events[:, -1]

    X = epochs.get_data()  # format is in (trials, channels, samples)
    X = X.astype(np.double)
    y = labels
    num_train = len(X)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    sampler = SubsetRandomSampler(indices)

    train_data = []
    for i in range(len(X)):
        train_data.append([X[i], y[i]])

    print("X shape: {}".format(X.shape))
    print("Y shape: {}".format(y.shape))

    batch_size = 64  # how many samples per batch to load
    num_workers = 0

    # loader = P300_IIB(train_data)
    loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                         sampler=sampler, num_workers=num_workers)

    return loader

def build_dataloader(path):
    logger.info("loading data from path: {}".format(path))
    train_loader = load_folder('train', path)
    valid_loader = load_folder('valid', path)

    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print(type(images))
    print(images.shape)
    print(labels.shape)

    return train_loader, valid_loader


def train_shapes(loader):
    data, labels = iter(loader).next()
    print("data shape: {}".format(data.shape))
    print("labels shape:{}".format(labels.shape))


def add_marker_choose_columns(channels_to_include, df):
    df = df.astype(float)
    df['Marker'] = df.apply(lambda row: label_marker(row), axis=1)
    df = df[['samplenr'] + channels_to_include + ['Marker']]
    return df


class P300_IIB(Dataset):
    def __init__(self,
                 data):
        self.data = data
        self.len = len(data)

    def __getitem__(self, index):
        sample = self.data[index][0]
        label = self.data[index][1]
        sample = np.reshape(sample, (1, sample.shape[0], sample.shape[1]))
        return sample, label

    def __len__(self):
        return self.len
