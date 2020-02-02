import torch
import json

from nets.two_layers_net import TwoLayersNet


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    return device


def load_conf():
    import yaml

    with open('conf.yaml', 'r', newline='') as f:
        conf = yaml.load(f)
        return \
            conf['data_dir'], \
            conf['n_epochs']


def getModel(device):
    model = TwoLayersNet().double()
    model.to(device)
    return model


# new code
from collections import OrderedDict

import torch
import yaml

from nets.two_layers_net import TwoLayersNet

_reject = {'eeg': 100e-4}

first_4_columns = [a for a in range(0, 4)]
first_4_col_names = {
    'sig_1': 'fc5',
    'sig_2': 'fc3',
    'sig_3': 'fc1',
    'sig_4': 'fcz',
}

muse_cols = [22, 24, 43, 44]
muse_cols_names = {
    'sig_22': 'fp1',
    'sig_24': 'fp2',
    'sig_43': 't9',
    'sig_44': 't10',
}

_conditions = OrderedDict()
_conditions['Non-target'] = [1]
_conditions['Target'] = [2]
_event_ids = {'Non-Target': 1, 'Target': 2}


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    return device


def load_conf_preprocess():
    with open('conf_prepross.yaml', 'r', newline='') as f:
        conf = yaml.load(f)
        return \
            conf['data_input'], \
            conf['data_output']


def load_conf_train():
    with open('conf_train.yaml', 'r', newline='') as f:
        conf = yaml.load(f)
        return \
            conf['input_path'], \
            conf['n_epochs'], \
            conf['model_name']


def load_conf_eval():
    with open('conf_eval.yaml', 'r', newline='') as f:
        conf = yaml.load(f)
        return \
            conf['data_dir'], \
            conf['model_dir']


def build_model(device):
    model = TwoLayersNet().double()
    model.to(device)
    return model
