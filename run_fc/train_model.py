import torch
import numpy as np
import matplotlib.pyplot as plt


def train_model(model, n_epochs, train_loader,valid_loader, device, optimizer, criterion):
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
            data = data.to(device)
            target = target.to(device)

            # TODO HOW TO REMOVE?
            shape = target.shape[0]
            target = target.reshape(shape, 1)
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
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available

            shape = target.shape[0]
            target = target.reshape(shape, 1)

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
            torch.save(model.state_dict(), 'model_cifar_new_4.pt')
            valid_loss_min = valid_loss

        # Plot the resulting loss over time
    plt.plot(tLoss, label='Training Loss')
    plt.plot(vLoss, label='Validation Loss')
    plt.legend()
    plt.show()
