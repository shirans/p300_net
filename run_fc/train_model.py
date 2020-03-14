import os
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


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
