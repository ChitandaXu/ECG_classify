import numpy as np
import pywt

from ecg_classify.features import Kur, Skew, MinDiff, Slope, RWidth
from ecg_classify.utils import denoise
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
    wave_start = (sample - 75).astype(int)
    wave_end = (sample + 75).astype(int)
    r_start_50 = (sample - 50).astype(int)
    r_end_50 = (sample + 50).astype(int)
    slope_start = sample.astype(int)
    slope_end = (sample + 50).astype(int)

    # rescale:
    rescale = __compute_rescale_x(num)
    pre_rr = pre_rr / rescale
    post_rr = post_rr / rescale

    # compute wavelet feature
    sig = read_sig(num)
    # cA4, cD4, cD3, cD2, cD1 = get_wavelet(sig, wave_start, wave_end)

    # compute morph feature
    clean_sig = denoise(sig)
    p_kur = get_morph(clean_sig, p_start, p_end, Kur())
    p_skew = get_morph(clean_sig, p_start, p_end, Skew())
    t_kur = get_morph(clean_sig, t_start, t_end, Kur())
    t_skew = get_morph(clean_sig, t_start, t_end, Skew())
    r_kur = get_morph(clean_sig, r_start, r_end, Kur())
    r_skew = get_morph(clean_sig, r_start, r_end, Skew())
    min_diff = get_morph(clean_sig, r_start, r_end, MinDiff())
    r_width = get_morph(clean_sig, r_start_50, r_end_50, RWidth())
    slope = get_morph(clean_sig, slope_start, slope_end, Slope())

    morph = __triple_feature(
        pre_rr, post_rr, p_kur, p_skew, t_kur, t_skew, r_kur, r_skew, min_diff, r_width, slope)

    front = morph[0]
    cur = morph[1]
    back = morph[2]
    # wavelet = __trim_feature(cA4, cD4, cD3, cD2, cD1)
    # front = np.hstack((morph[0], wavelet[0]))
    # cur = np.hstack((morph[1], wavelet[1]))
    # back = np.hstack((morph[2], wavelet[2]))

    # 4nd -> (n-1)th
    symbols = np.expand_dims(symbol[1: -1], 1)

    return np.hstack([front, cur, back, symbols])


def __triple_feature(*args):
    if np.ndim(args[0]) == 1:
        res = np.vstack(args).transpose()
    else:
        res = np.hstack(args)
    front = res[: -2]
    cur = res[1: -1]
    back = res[2:]
    return front, cur, back


def __compute_rescale(num):
    symbol = read_symbol(num)
    sample = read_sample(num)
    rr = sample[1:] - sample[:-1]
    pre_rr = rr[:-1]
    symbol = symbol[1:-1]
    normal = pre_rr[symbol == 'N']
    if len(normal) == 0:
        return 1
    avg = sum(normal) / len(normal)
    return avg / 300


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



