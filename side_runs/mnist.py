import numpy as np
from torch import optim, nn
from dataloaders import mnist
from dataloaders.mnist import MNIST
from nets.lenet import LeNet
from nets.twolayersnet import TwoLayersNet
from preprocess.helpers import get_device
from run_fc.eval_model import evaluate
from run_fc.metadata import Metadata
from run_fc.ready_made_models import train_svm
from run_fc.train_model import train_model, build_model_path


def deep_net(train_loader,valid_loader):
    # model = TwoLayersNet(10)
    model = LeNet(10)

    device = get_device()
    # Get the device

    # specify loss function
    criterion = nn.CrossEntropyLoss()

    # specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    output_path, info_path = build_model_path('lenet_2')

    valid_loss_min = np.Inf  # track change in validation loss
    metadata = Metadata(info_path)
    metadata.append(str(model))
    stop_condition = {'patience': 20, 'epsilon': 0.00001}
    train_model(model=model, n_epochs=100, train_loader=train_loader, valid_loader=valid_loader, device=device,
                optimizer=optimizer, criterion=criterion, stop_condition=stop_condition, output_path=output_path,
                metadata=metadata)

    evaluate(train_loader, device, model, metadata)
    evaluate(valid_loader, device, model, metadata)


(x_train,y_train), (x_valid,y_valid) = mnist.build_numpy('/Users/shiran.s/dev/p300_net/data/mnist/small',True)
x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])
x_valid = x_valid.reshape(-1,x_valid.shape[1]*x_valid.shape[2])
train_svm(x_train, y_train, x_valid, y_valid, Metadata(None))
