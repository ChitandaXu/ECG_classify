import numpy as np

from ecg_classify.utils import denoise
from ecg_classify.wfdb_io import read_sample, read_symbol, read_sig

# num_list = [100, 101, 103, 105, 106, 108, 109, 111, 112, 116, 118, 119, 124, 200, 201, 202, 203, 205, 207, 208, 209,
#             212, 213, 214, 215, 219, 220, 221, 222, 223, 228, 231, 232]


def gen_sample(num):
    sig = read_sig(num)
    sig = denoise(sig)
    sample = read_sample(num)
    symbol = read_symbol(num)
    sample = sample[2: -1]
    symbol = symbol[2: -1]
    num_of_r_wave = len(sample)
    beats = np.empty((num_of_r_wave, 300))
    for i in range(num_of_r_wave):
        start = sample[i] - 150
        end = sample[i] + 150
        if start < 0 or end > 650000:
            raise Exception('start point or end point is out of range.')
        beats[i] = sig[start: end]
    return beats, symbol


def gen_feature(num):
    sig = read_sig(num)
    sig = denoise(sig)
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

    # compute rescale coefficient
    rescale_x = __compute_rescale_x(num)

    # compute morph feature
    p_kur, p_skew = __compute_morph(sig, p_start, p_end)
    t_kur, t_skew = __compute_morph(sig, t_start, t_end)
    r_kur, r_skew = __compute_morph(sig, r_start, r_end)

    pre_rr_f, post_rr_f, p_kur_f, p_skew_f, t_kur_f, t_skew_f, r_kur_f, r_skew_f = \
        list(map(lambda x: x[:-1], [pre_rr, post_rr, p_kur, p_skew, t_kur, t_skew, r_kur, r_skew]))
    pre_rr_m, post_rr_m, p_kur_m, p_skew_m, t_kur_m, t_skew_m, r_kur_m, r_skew_m, symbol = \
        list(map(lambda x: x[1:], [pre_rr, post_rr, p_kur, p_skew, t_kur, t_skew, r_kur, r_skew, symbol]))

    # rescale:
    pre_rr_m = pre_rr_m / rescale_x
    post_rr_m = post_rr_m / rescale_x
    pre_rr_f = pre_rr_f / rescale_x
    post_rr_f = post_rr_f / rescale_x
    # return np.array([pre_rr_m, post_rr_m, p_kur_m, p_skew_m, t_kur_m, t_skew_m, r_kur_m, r_skew_m,
    #                  symbol]).transpose()
    return np.array([pre_rr_m, post_rr_m, p_kur_m, p_skew_m, t_kur_m, t_skew_m, r_kur_m, r_skew_m,
                     pre_rr_f, post_rr_f, p_kur_f, p_skew_f, t_kur_f, t_skew_f, r_kur_f, r_skew_f,
                     symbol]).transpose()


def __compute_rescale_y(sig):
    y_sum = sum(abs(sig))
    return y_sum / 200000


def __compute_rescale_x(num):
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


