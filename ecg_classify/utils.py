import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy import signal

from ecg_classify.wfdb_io import read_sig, read_symbol, read_sample


def plot_hbs(num, symbol, count=1, width=150, need_filter=False, merge=False):
    beats, symbols = gen_sample(num, width, need_filter)
    sig = beats[symbols == symbol]
    n = sig.shape[0]
    count = min(count, n)
    if merge:
        plt.figure()
    for i in range(count):
        if not merge:
            plt.figure()
        plt.plot(sig[i])


def plot_hb(num, symbol, width=150, need_filter=False):
    beats, symbols = gen_sample(num, width, need_filter)
    sig = beats[symbols == symbol]
    cur = sig[2]
    np.savetxt('a.csv', cur, delimiter=',')
    plt.figure()
    plt.plot(cur)


def flip_sig(sig):
    return -sig


def gen_sample(num, width=150, need_filter=False):
    sig = read_sig(num)
    if need_filter:
        sig = bandpass(sig, 30)
        # sig = wav(sig)
    sample = read_sample(num)
    symbols = read_symbol(num)
    sample = sample[3: -3]
    symbols = symbols[3: -3]
    num_of_r_wave = len(sample)
    beats = np.empty((num_of_r_wave, width * 2))
    for i in range(num_of_r_wave):
        start = sample[i] - width
        end = sample[i] + width
        if start < 0 or end > 650000:
            raise Exception('start point or end point is out of range.')
        beats[i] = sig[start: end]
    return beats, symbols


def bandpass(sig, ntaps, f1=0.5/360, f2=70/360):
    sig = np.expand_dims(sig, 0)
    b = signal.firwin(ntaps, [f1, f2], pass_zero=False)
    y = signal.filtfilt(b, a=[1], x=sig)
    # conv_result = signal.convolve(sig, b[np.newaxis, :], mode='valid')
    return y[0]


def wav(sig):
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


def count_num(num, label):
    symbols = read_symbol(num)[3: -3]  # 4nd -> (n-3)th 数量和 feature 中的样本保持一致
    res = len(symbols[symbols == label])
    return res


def count_num_by_ds(data_set, label):
    count = 0
    for num in data_set:
        count += count_num(num, label)
    return count
