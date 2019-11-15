import numpy as np


def shuffle_data(x, y):
    if x.shape[0] != y.shape[0]:
        raise Exception("Invalid input, x and y should be same length in dimension 0")
    # np.random.seed(7)
    order = np.random.permutation(np.arange(x.shape[0]))
    x = x[order]
    y = y[order]
    return x, y
