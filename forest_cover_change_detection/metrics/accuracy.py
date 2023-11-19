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
