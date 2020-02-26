import os
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from preprocess.preprocess import load_train_valid_matrix, take_one_sample, take_one_sample_per_class

warnings.filterwarnings('ignore')


def train_svm(data_dit):
    x_train, y_train, x_valid, y_valid = load_train_valid_matrix(data_dit, None)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_valid = x_valid.reshape(x_valid.shape[0], -1)

    from sklearn.svm import SVC
    clf = SVC(gamma='auto', kernel='linear')
    clf.fit(x_train, y_train)
    eval_read_maid_models(clf, x_valid, y_valid, 'svm', x_train, y_train)

    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)

    tree.plot_tree(clf)
    eval_read_maid_models(clf, x_valid, y_valid, 'tree', x_train, y_train)


def eval_read_maid_models(clf, x_valid, y_valid, model_name, x_train, y_train):
    eval_read_inner(clf,x_train,y_train,model_name,'train')
    eval_read_inner(clf,x_valid,y_valid,model_name,'valud')


def eval_read_inner(clf, x, y, model_name,data_type):
    pred = clf.predict(x)
    bincount = np.bincount(pred)
    prediction_right = pred == y
    correct = np.sum(prediction_right)
    total = len(pred)
    print("----------------------------------------------")
    print("data_type: ",data_type)
    print("bincount per class: ", bincount)
    print("correct:", correct, "total: ", total)
    print("score for ", model_name, ":", accuracy_score(y, pred))
    print("balanced score for ", model_name, ":", balanced_accuracy_score(y, pred))
    print("----------------------------------------------")


def train_model(model, n_epochs, train_loader, valid_loader, device, optimizer, criterion, model_name):
    valid_loss_min = np.Inf
    tLoss, vLoss = [], []

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
            data, target = data.to(device), target.to(device)

            # print("data shape:", data.shape)
            # print("target shape:", target.shape)
            # TODO HOW TO REMOVE?
            # shape = target.shape[0]
            # target = target.reshape(shape, 1)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            # print("tarhe sja[e",target.shape)
            # print("output sja[e",output.shape)
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
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            output_path = build_model_path(model_name)
            torch.save(model.state_dict(), output_path)
            print('saved model to path:  {}'.format(output_path))
            valid_loss_min = valid_loss

    # Plot the resulting loss over time
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


def build_model_path(model_name):
    folder = build_folder_path(model_name)
    output_path = os.path.join(folder, "model.pt")
    return output_path


def build_folder_path(model_name):
    folder = os.path.join("/Users/shiran.s/dev/p300_net/output/models/", model_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder
