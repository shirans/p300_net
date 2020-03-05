from collections import OrderedDict
import nets.all_nets

import torch
import yaml

from side_runs.dataloader import MnistNet

replace_ch_names = {'1': "TP9"}

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

conditions = OrderedDict()
conditions['Non-target'] = [1]
conditions['Target'] = [2]
event_ids = {'Non-Target': 1, 'Target': 2}


def get_model(device, network_class):
    # model =  nets.all_nets.TwoLayersNet().double()
    # return model
    # model = MnistNet().double()
    print("network used:", network_class)
    model = getattr(nets.all_nets, network_class)().double()
    model.to(device)
    return model


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    return device


def load_conf_preprocess():
    with open('config/conf_preprocess.yaml', 'r', newline='') as f:
        conf = yaml.load(f)
        return \
            conf['data_input'], \
            conf['data_output']


def conf():
    with open('config/conf_train.yaml', 'r', newline='') as f:
        conf = yaml.load(f)
        return conf


def load_conf_train():
    with open('config/conf_train.yaml', 'r', newline='') as f:
        conf = yaml.load(f)
        return \
            conf['data_dir'], \
            conf['n_epochs'], \
            conf['model_name'], \
            conf['network'], \
            conf['stop_condition']


def load_conf_eval():
    with open('config/conf_eval.yaml', 'r', newline='') as f:
        conf = yaml.load(f)
        return \
            conf['data_dir'], \
            conf['model_dir'], \
            conf['network']
