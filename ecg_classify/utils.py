import numpy as np
import pywt


def shuffle_data(x, y):
    if x.shape[0] != y.shape[0]:
        raise Exception("Invalid input, x and y should be same length in dimension 0")
    # np.random.seed(7)
    order = np.random.permutation(np.arange(x.shape[0]))
    x = x[order]
    y = y[order]
    return x, y


def denoise(sig):
    """
    Denoise by db6 with level 6

    :param signal: ECG signal
    :return: signal after denoise
    """
    coeffs = pywt.wavedec(sig, 'db6', level=6)
    cA6, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # only remain cD6 -> cD3
    cA6 = np.zeros(cA6.shape)
    cD2 = np.zeros(cD2.shape)
    cD1 = np.zeros(cD1.shape)
    coeffs = [cA6, cD6, cD5, cD4, cD3, cD2, cD1]
    return pywt.waverec(coeffs, 'db6')
