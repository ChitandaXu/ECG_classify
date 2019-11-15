import numpy as np
import pywt

from ecg_classify.wfdb_io import read_sample, read_symbol, read_sig

# num_list = [100, 101, 103, 105, 106, 108, 109, 111, 112, 116, 118, 119, 124, 200, 201, 202, 203, 205, 207, 208, 209,
#             212, 213, 214, 215, 219, 220, 221, 222, 223, 228, 231, 232]


def gen_feature(num, rescale=True):
    sig = read_sig(num)
    sig = __denoise(sig)
    symbol = read_symbol(num)
    sample = read_sample(num)
    rr = sample[1:] - sample[:-1]
    pre_rr = rr[:-1]
    post_rr = rr[1:]
    sample = sample[1:-1]
    symbol = symbol[1:-1]
    p_start = (sample - pre_rr * 0.35).astype(int)
    p_end = (sample - pre_rr * 0.05).astype(int)
    t_start = (sample + post_rr * 0.05).astype(int)
    t_end = (sample + post_rr * 0.65).astype(int)
    r_start = (sample - 22).astype(int)
    r_end = (sample + 22).astype(int)
    rescale_coefficient = __compute_rescale_coefficient(num)
    p_kur, p_skew = __compute_morph(sig, p_start, p_end)
    t_kur, t_skew = __compute_morph(sig, t_start, t_end)
    r_kur, r_skew = __compute_morph(sig, r_start, r_end)
    if rescale:
        pre_rr = pre_rr / rescale_coefficient
        post_rr = post_rr / rescale_coefficient
    return np.array([pre_rr, post_rr, p_kur, p_skew, t_kur, t_skew, r_kur, r_skew, symbol]).transpose()


def __compute_rescale_coefficient(num):
    signal = read_sig(num)
    samples = read_sample(num)
    rr = samples[1:] - samples[:-1]
    avg = sum(rr) / rr.shape[0]
    return avg / 300


def __compute_morph(sig, start, end):
    res_size = start.shape[0]
    kur = np.zeros(res_size)
    skew = np.zeros(res_size)
    for idx in range(res_size):
        cur_sig = sig[start[idx]: end[idx]]
        mean = np.mean(cur_sig)
        std_val = np.std(cur_sig, ddof=1)
        n = len(cur_sig)
        kur[idx] = 1 / (n - 1) * np.sum((cur_sig - mean) ** 4)
        skew[idx] = kur[idx] / (std_val ** 3)
    return np.array([kur, skew])


def __denoise(sig):
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


