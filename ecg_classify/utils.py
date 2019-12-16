import numpy as np
import pywt
import matplotlib.pyplot as plt

from ecg_classify.wfdb_io import read_sig, read_symbol, read_sample


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


def plot_hb(num, symbol, count):
    beats, symbols = gen_sample(num)
    sig = beats[symbols == symbol]
    n = sig.shape[0]
    count = min(count, n)
    plt.figure()
    for i in range(count):
        plt.plot(sig[i])


def gen_sample(num):
    sig = read_sig(num)
    sig = denoise(sig)
    sample = read_sample(num)
    symbols = read_symbol(num)
    sample = sample[3: -3]
    symbols = symbols[3: -3]
    num_of_r_wave = len(sample)
    beats = np.empty((num_of_r_wave, 300))
    for i in range(num_of_r_wave):
        start = sample[i] - 150
        end = sample[i] + 150
        if start < 0 or end > 650000:
            raise Exception('start point or end point is out of range.')
        beats[i] = sig[start: end]
    return beats, symbols
