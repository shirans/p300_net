import torch

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
