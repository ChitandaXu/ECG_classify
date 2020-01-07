import numpy as np
import pywt

from ecg_classify.features import Kur, Skew, MinDiff, Slope, QrsWidth, PRInterval
from ecg_classify.utils import wav, bandpass
from ecg_classify.wfdb_io import read_sample, read_symbol, read_sig


def gen_feature(num):
    symbol = read_symbol(num)
    sample = read_sample(num)
    rr = sample[1:] - sample[:-1]
    pre_rr = rr[1:-2]
    post_rr = rr[2:-1]
    sample = sample[2:-2]
    symbol = symbol[2:-2]

    p_start = (sample - pre_rr * 0.35).astype(int)
    p_end = (sample - pre_rr * 0.05).astype(int)
    t_start = (sample + post_rr * 0.05).astype(int)
    t_end = (sample + post_rr * 0.65).astype(int)
    r_start = (sample - 30).astype(int)
    r_end = (sample + 30).astype(int)
    slope_start = sample.astype(int)
    slope_end = (sample + 50).astype(int)
    pr_start = (sample - 120).astype(int)
    pr_end = sample.astype(int)
    amp_start = (sample - 80).astype(int)
    amp_end = (sample + 80).astype(int)

    # rescale:
    rescale = __compute_rescale_x(num)
    pre_rr = pre_rr / rescale

    # compute morph feature
    sig = read_sig(num)
    sig_b = bandpass(sig, 30)
    sig_w = wav(sig)
    p_kur = get_morph(sig, p_start, p_end, Kur())
    p_skew = get_morph(sig, p_start, p_end, Skew())
    t_kur = get_morph(sig, t_start, t_end, Kur())
    t_skew = get_morph(sig, t_start, t_end, Skew())
    r_kur = get_morph(sig, r_start, r_end, Kur())
    r_skew = get_morph(sig, r_start, r_end, Skew())
    min_diff = get_morph(sig_w, r_start, r_end, MinDiff())  # w
    slope = get_morph(sig_b, slope_start, slope_end, Slope())  # b
    pr_interval = get_morph(sig_b, pr_start, pr_end, PRInterval())  # b
    r_width = get_morph(sig_b, amp_start, amp_end, QrsWidth())  # b

    morph = __triple_feature(
        pre_rr, p_kur, p_skew, t_kur, t_skew, r_kur, r_skew, min_diff, r_width, slope)
    other = __trim_feature(pr_interval)

    front = morph[0]
    cur = morph[1]
    back = morph[2]

    # 4nd -> (n-1)th
    symbols = np.expand_dims(symbol[1: -1], 1)
    features = np.hstack([front, cur, back, other, symbols])
    return features


def __trim_feature(*args):
    if np.ndim(args[0]) == 1:
        res = np.vstack(args).transpose()
    else:
        res = np.hstack(args)
    cur = res[1: -1]
    return cur


def __triple_feature(*args):
    if np.ndim(args[0]) == 1:
        res = np.vstack(args).transpose()
    else:
        res = np.hstack(args)
    front = res[: -2]
    cur = res[1: -1]
    back = res[2:]
    return front, cur, back


def __compute_rescale_x(num):
    samples = read_sample(num)
    rr = samples[1:] - samples[:-1]
    avg = sum(rr) / rr.shape[0]
    return avg / 300


def get_morph(sig, start, end, feature):
    n = start.shape[0]
    morph = np.zeros(n)
    for i in range(n):
        lo = start[i]
        hi = end[i]
        morph[i] = feature.execute(sig, lo, hi)
    return morph


def __generate_wavelet_array(sig, size):
    [a4, d4, d3, d2, d1] = pywt.wavedec(sig, 'db4', level=4)
    cA4 = np.empty((size, len(a4)))
    cD4 = np.empty((size, len(d4)))
    cD3 = np.empty((size, len(d3)))
    cD2 = np.empty((size, len(d2)))
    cD1 = np.empty((size, len(d1)))
    return cA4, cD4, cD3, cD2, cD1


def get_wavelet(sig, start, end):
    res_size = start.shape[0]
    cA4, cD4, cD3, cD2, cD1 = __generate_wavelet_array(sig[start[0]: end[0]], res_size)

    for idx in range(res_size):
        cur_sig = sig[start[idx]: end[idx]]
        [cA4[idx], cD4[idx], cD3[idx], cD2[idx], cD1[idx]] = pywt.wavedec(cur_sig, 'db4', level=4)
    return cA4, cD4, cD3, cD2, cD1
