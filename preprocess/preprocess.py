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

from plot_p300 import plot_p300, plot_inner
from preprocess.helpers import replace_ch_names, _reject, event_ids, conditions

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# preprocess - load the data, split to train,test, set lables
# Digitized to 240Mz, we want 600 MS to identify P300 at the half of it so 250 samples are 600 MS

MEAN = None


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
        return [data_dict_to_df(matrix, indexes)]
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


def take_small_sample(X, labels, model_type):
    if model_type == 'train':
        X = X[:24, :]
        labels = labels[:24]
    return X, labels


def take_one_sample_per_class(X, labels, model_type):
    if model_type == 'train':
        X = np.array([X[0], X[2]])
        labels = np.array([0, 1])
    return X, labels


def map_classes(label):
    # map 1 -> 0, 2-->1
    return label - 1


map_classes_virtual = np.vectorize(map_classes)


def calculateEntropy(dataSet):
    number = len(dataSet)
    labelCounts = {}
    for label in dataSet:
        if label not in labelCounts.keys():
            labelCounts[label] = 0
        labelCounts[label] += 1
    entropy = 0
    for i in labelCounts:
        probability = float(labelCounts[i]) / number
        entropy -= probability * np.math.log(probability, 2)
    return entropy


def load_folder(model_type, path, batch_size, sample_example, metadata, mean_substract):
    x, labels = load_x_labels(model_type, path, sample_example, mean_substract)
    print("entropy for {}:{}".format(model_type,calculateEntropy(labels)))
    # X, labels = take_one_sample(X, labels, model_type)
    # X, labels = take_one_sample_per_class(X, labels, model_type)
    # X, lables = take_small_sample(X, labels, model_type)
    num_train = len(x)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    sampler = SubsetRandomSampler(indices)

    train_data = []
    for i in range(len(x)):
        train_data.append([x[i], labels[i]])

    metadata.append("X shape: {} for model type {}".format(x.shape, model_type))
    metadata.append("Y shape: {} for model type {}".format(labels.shape, model_type))

    num_workers = 0

    train_data = P300_IIB(train_data)
    loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                         sampler=sampler, num_workers=num_workers)

    return loader


def load_x_labels(model_type, path, sample_example, mean_substract):
    full_path = os.path.join(path, model_type + '.csv')
    num_columns_in_file = len(pd.read_csv(full_path, index_col=0).columns)
    # all columns are eeg except the last one which is the marker
    ch_ind = list(range(0, num_columns_in_file))[:-1]
    raw = load_muse_csv_as_raw__copy([full_path], sfreq=240, stim_ind=-1, replace_ch_names=replace_ch_names,
                                     ch_ind=ch_ind)
    raw.filter(1, 30, method='iir')
    events = find_events(raw)
    epochs = mne.Epochs(raw, events=events, event_id=event_ids,
                        tmin=-0.1, tmax=0.8, baseline=None,
                        reject=_reject, preload=True,
                        verbose=False, picks=ch_ind)
    print('sample drop for model type ', model_type, ' %', (1 - len(epochs.events) / len(events)) * 100)
    # plot_inner(epochs,conditions,('in code mode type: ' + model_type),ylim=(-100, 100))
    labels = epochs.events[:, -1]
    labels = map_classes_virtual(labels)
    # x = epochs.get_data()  # format is in (trials, channels, samples)
    x = epochs.get_data() * 1e6
    x = x.astype(np.double)
    global MEAN
    if mean_substract:
        if model_type == 'train':
            MEAN = np.mean(x, axis=0)
        if MEAN is None:
            raise ("error - mean was not computed")
        x = x - MEAN

    if sample_example is not None:
        print("samling data!!!!!!!!!")
        x, labels = sample_example(x, labels, model_type)

    print("labels:", np.bincount(labels), ' for:', model_type)
    return x, labels


# to test why it always false on class 2, create DS with only one item of class 2
def take_one_sample(X, labels, model_type):
    if model_type == 'train':
        X = np.array([X[2]])
        labels = np.array([0])
    return X, labels


def load_train_valid_matrix(path, sample, mean_substract):
    x_train, y_train = load_x_labels('train', path, sample, mean_substract)
    x_valid, y_valid = load_x_labels('valid', path, sample, mean_substract)
    return x_train, y_train, x_valid, y_valid


def build_dataloader(path, batch_size, metadata, mean_substract):
    logger.info("loading data from path: {}".format(path))
    train_loader = load_folder('train', path, batch_size, None, metadata, mean_substract)
    valid_loader = load_folder('valid', path, batch_size, None, metadata, mean_substract)

    images, labels = iter(train_loader).next()
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
        sample = sample.reshape(1, sample.shape[0], sample.shape[1])
        return sample, label

    def __len__(self):
        return self.len
