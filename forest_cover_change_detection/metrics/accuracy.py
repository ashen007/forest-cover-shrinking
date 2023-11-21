import torch
import numpy as np


def calculate_confusion(y_true, y_pred):
    tp = np.logical_and(y_pred, y_true).sum()
    tn = np.logical_and(np.logical_not(y_pred), np.logical_not(y_true)).sum()
    fp = np.logical_and(y_pred, np.logical_not(y_true)).sum()
    fn = np.logical_and(np.logical_not(y_pred), y_true).sum()

    return tp, tn, fp, fn


def pixel_accuracy(y_true, y_pred):
    tp, tn, fp, fn = calculate_confusion(y_true, y_pred)
    return (tp + tn) / (tp + tn + fp + fn)


def class_accuracy(y_true, logits):
    class_correct = [0. for _ in range(2)]
    class_total = [0. for _ in range(2)]
    class_accuracy = list(0. for _ in range(2))

    _, predicted = torch.max(logits.data, 0)
    gt = torch.unsqueeze(torch.from_numpy(1.0 * y_true.numpy()), 0).float()
    c = (predicted.int() == gt.data.int())

    for i in range(c.size(1)):
        for j in range(c.size(2)):
            l = int(gt.data[0, i, j])
            class_correct[l] += c[0, i, j]
            class_total[l] += 1

    for i in range(2):
        class_accuracy[i] = class_correct[i] / max(class_total[i], 0.00001)
        class_accuracy[i] = float(class_accuracy[i])

    return class_accuracy


def kappa(y_true, y_pred):
    tp, tn, fp, fn = calculate_confusion(y_true, y_pred)
    N = tp + tn + fp + fn
    p0 = (tp + tn) / N
    pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (N * N)

    return (p0 - pe) / (1 - pe)


def precision(y_true, y_pred):
    tp, tn, fp, fn = calculate_confusion(y_true, y_pred)

    return tp / (tp + fp)


def recall(y_true, y_pred):
    tp, tn, fp, fn = calculate_confusion(y_true, y_pred)

    return tp / (tp + fn)


def dice(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)

    return 2 * prec * rec / (prec + rec)
