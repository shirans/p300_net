import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

from run_fc.train_model import metrics_success

NUM_CLASSES = 2


def evaluate(loader, device, model, type, metadata):
    model.eval()
    # track test loss
    test_loss = 0.0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    criterion = nn.CrossEntropyLoss()

    all_outputs = []
    all_targets = []
    for data, target in loader:
        # move tensors to GPU if CUDA is available
        data, target = data.to(device), target.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        all_outputs.append(pred.detach().numpy())
        all_targets.append(target.detach().numpy())
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        if len(correct_tensor) > 1:
            correct = np.squeeze(correct_tensor.numpy()) if device == "cpu" else np.squeeze(
                correct_tensor.cpu().numpy())
        else:
            correct = correct_tensor.numpy()
        # calculate test accuracy for each object class
        for i in range(target.size(0)):
            label = target.data[i]
            if label > 1:
                print(label)
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    pred = np.concatenate(all_outputs)
    y = np.concatenate(all_targets)
    metrics_success("deep model", y, pred, metadata)
    # average test loss
    test_loss = test_loss / len(loader.dataset)

    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            metadata.append('Accuracy of %3s: %2d%% (%2d/%2d)' % (
                i, 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            metadata.append('Accuracy of %3s: N/A (no training examples)' % i)

    metadata.append('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
