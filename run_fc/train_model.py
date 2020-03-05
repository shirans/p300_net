import os
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from preprocess.preprocess import load_train_valid_matrix, take_one_sample, take_one_sample_per_class

warnings.filterwarnings('ignore')


def train_svm(data_dit, mean_substract, metadata):
    x_train, y_train, x_valid, y_valid = load_train_valid_matrix(data_dit, None, mean_substract)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_valid = x_valid.reshape(x_valid.shape[0], -1)

    from sklearn.svm import SVC
    clf = SVC(gamma='auto', kernel='linear')
    clf.fit(x_train, y_train)
    eval_read_maid_models(clf, x_valid, y_valid, 'svm', x_train, y_train, metadata)

    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)

    tree.plot_tree(clf)
    eval_read_maid_models(clf, x_valid, y_valid, 'tree', x_train, y_train)


def eval_read_maid_models(clf, x_valid, y_valid, model_name, x_train, y_train, metadata):
    eval_read_inner(clf, x_train, y_train, model_name + 'train', metadata)
    eval_read_inner(clf, x_valid, y_valid, model_name + 'test', metadata)


def eval_read_inner(clf, x, y, model_name, metadata):
    pred = clf.predict(x)
    metrics_success(model_name, y, pred, metadata)


def metrics_success(model_name, y, pred, metadata):
    prediction_right = pred == y
    correct = np.sum(prediction_right)
    total = len(pred)
    np.bincount(y)
    metadata.append("-----------------success metrics: {}-----------------".format(model_name))
    metadata.append("correct: {} total: {}".format(correct, total))
    metadata.append("accuracy: {}".format( accuracy_score(y, pred)))
    metadata.append("balanced accuracy: {}".format(balanced_accuracy_score(y, pred)))
    metadata.append(classification_report(y, pred))
    metadata.append("--------------------------------------------------")


def train_model(model, n_epochs, train_loader, valid_loader, device, optimizer, criterion, stop_condition, output_path,
                metadata):
    valid_loss_min = np.Inf
    tLoss, vLoss = [], []
    patience = stop_condition['patience']
    epsilon = stop_condition['epsilon']
    num_declaing_epochs = 0
    num_epochs = 0

    for epoch in range(n_epochs):
        num_epochs = num_epochs + 1
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        #########
        # train #
        #########
        model.train()
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            data, target = data.to(device), target.to(device)
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
        model.train(False)
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
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
        if epoch % 10 == 0:
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min and abs(valid_loss_min - valid_loss) < epsilon:
            print("loss decreased but improvement is small: {} - {} = {} > {}",
                  valid_loss_min, valid_loss, valid_loss_min - valid_loss, epsilon)
        if valid_loss <= valid_loss_min and abs(valid_loss_min - valid_loss) >= epsilon:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), output_path)
            print('saved model to path:  {}'.format(output_path))
            valid_loss_min = valid_loss
            num_declaing_epochs = 0
            print("patience counter reset. loss:", valid_loss_min, "num epochs:", num_epochs)
        else:
            num_declaing_epochs = num_declaing_epochs + 1
        if num_declaing_epochs > patience:
            print("stopping after {} non improving epochs. num total epochs {}".format(num_declaing_epochs, num_epochs))
            break

    # Plot the resulting loss over time
    metadata.append("num epochs:".format(num_epochs))
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_ylabel('Training Loss')
    ax1.set_xlabel('loss')
    ax1.plot(tLoss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Validation Loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(vLoss, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    metadata.addfig(fig)


def build_model_path(model_name):
    folder = build_folder_path(model_name)
    output_path = os.path.join(folder, "model.pt")
    return output_path, folder


def build_folder_path(model_name):
    folder = os.path.join("/Users/shiran.s/dev/p300_net/output/models/", model_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder
