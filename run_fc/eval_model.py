import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report


def evaluate(loader, device, model, metadata):
    model.eval()
    all_outputs = []
    all_targets = []
    for data, target in loader:
        # move tensors to GPU if CUDA is available
        data, target = data.to(device), target.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        all_outputs.append(pred.detach().numpy())
        all_targets.append(target.detach().numpy())
        # compare predictions to true label

    prediction = np.concatenate(all_outputs)
    y = np.concatenate(all_targets)
    metrics_success("deep model", y, prediction, metadata)


def metrics_success(model_name, y, pred, metadata):
    prediction_right = pred == y
    correct = np.sum(prediction_right)
    total = len(pred)
    metadata.append("-----------------success metrics: {}-----------------".format(model_name))
    metadata.append("correct: {} total: {}".format(correct, total))
    metadata.append("accuracy: {}".format(accuracy_score(y, pred)))
    metadata.append("balanced accuracy: {}".format(balanced_accuracy_score(y, pred)))
    metadata.append("\n{}".format(classification_report(y, pred)))
    metadata.append("--------------------------------------------------")
