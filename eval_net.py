import torch
import torch.nn as nn
import numpy as np
from copy_helpers import load_data, load_muse_csv_as_raw__copy
from nets.two_layers_net import TwoLayersNet
from preprocess import preprocess
from preprocess.helpers import load_conf, get_device

model = TwoLayersNet().double()
model.load_state_dict(torch.load('model_cifar_new_2.pt'))
model.eval()

data_dir, n_epochs = load_conf()
device = get_device()
# raws = load_muse_csv_as_raw__copy(filename=[data_dir], replace_ch_names={'1': 'tp9'})
train_loader, valid_loader = preprocess.load_data(data_dir)

NUM_CLASSES = 3

# track test loss
test_loss = 0.0
class_correct = [0] * NUM_CLASSES
class_total = [0] * NUM_CLASSES

criterion = nn.CrossEntropyLoss()
# For generating confusion matrix

# iterate over test data
for data, target in valid_loader:
    # move tensors to GPU if CUDA is available
    data = data.to(device)
    target = target.to(device)

    # TODO HOW TO REMOVE?
    shape = target.shape[0]
    target = target.reshape(shape, 1)
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss
    test_loss += loss.item() * data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if device == "cpu" else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(target.size(0)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss / len(valid_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(NUM_CLASSES):
    if class_total[i] > 0:
        print('Test Accuracy of %3s: %2d%% (%2d/%2d)' % (
            i, 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %3s: N/A (no training examples)' % (class_total[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
