import numpy as np
import pywt


def shuffle_data(*args):
    # np.random.seed(7)
    x = args[0]
    order = np.random.permutation(np.arange(len(x)))
    for i in len(args):
        args[i] = args[i][order]
    return args


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
