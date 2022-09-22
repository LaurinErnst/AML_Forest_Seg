import numpy as np


def percentage_loss(x, y):
    x = np.flatten(x)
    y = np.flatten(y)

    x = np.round(x)
    y = np.round(y)

    return np.average(x == y)


def jaccard_index(mask_true, mask_pred):
    n_true = np.count_nonzero(mask_true == 1)
    n_pred = np.count_nonzero(mask_pred == 1)

    forest_true = mask_true == 1
    forest_pred = mask_pred == 1

    bool_array = forest_true * forest_pred
    n_cap = np.count_nonzero(bool_array)

    return n_cap / (n_true + n_pred - n_cap)
