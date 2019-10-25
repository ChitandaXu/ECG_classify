import numpy as np
import pywt
from functools import reduce
from ecg_classify.constants import DataSetType, label_list
from ecg_classify.wfdb_io import generate_sample_by_heartbeat, read_signal


def get_rr_interval(heartbeat_symbol, data_set_type=DataSetType.TRAINING):
    [data_set, r_loc_set, prev_r_loc_set, next_r_loc_set, number_set] = \
        generate_sample_by_heartbeat(heartbeat_symbol, data_set_type)
    anterior_rr_interval = r_loc_set - prev_r_loc_set
    posterior_rr_interval = next_r_loc_set - r_loc_set
    return np.array([anterior_rr_interval, posterior_rr_interval])


def get_p_feature(heartbeat_symbol, data_set_type=DataSetType.TRAINING):
    [data_set, r_loc_set, prev_r_loc_set, next_r_loc_set, number_set] = \
        generate_sample_by_heartbeat(heartbeat_symbol, data_set_type)

    # P region
    start = r_loc_set - ((r_loc_set - prev_r_loc_set) * 0.35).astype(int)
    end = r_loc_set - 22
    return calc_morph_feature(number_set, start, end, 'P')


def get_t_feature(heartbeat_symbol, data_set_type=DataSetType.TRAINING):
    [data_set, r_loc_set, prev_r_loc_set, next_r_loc_set, number_set] = \
        generate_sample_by_heartbeat(heartbeat_symbol, data_set_type)

    # T region
    start = r_loc_set + 22
    end = r_loc_set + ((next_r_loc_set - r_loc_set) * 0.65).astype(int)
    return calc_morph_feature(number_set, start, end, 'T')


def get_qrs_feature(heartbeat_symbol, data_set_type=DataSetType.TRAINING):
    [data_set, r_loc_set, prev_r_loc_set, next_r_loc_set, number_set] = \
        generate_sample_by_heartbeat(heartbeat_symbol, data_set_type)

    # QRS region
    start = r_loc_set - 22
    end = r_loc_set + 22
    return calc_morph_feature(number_set, start, end, 'QRS')


def calc_wavelet_feature(number_set, start, end):
    signal = np.zeros(650000)
    pre_val = -1
    for idx, val in enumerate(number_set):
        if val != pre_val:
            signal = read_signal(val)
            pre_val = val
        sig = signal[start[idx]: end[idx]]
        coeffs = pywt.wavedec(sig, 'db4', level=4)
        cA4, cD4, cD3, cD2, cD1 = coeffs
    return cA4, cD4, cD3, cD2


def calc_morph_feature(number_set, start, end, region):
    if region not in ['T', 'P', 'QRS']:
        raise Exception("region is invalid, please specify 'T' or 'P' or 'QRS'")
    signal = np.zeros(650000)
    pre_val = -1
    kurtosis = np.zeros(len(number_set))
    skewness = np.zeros(len(number_set))
    for idx, val in enumerate(number_set):
        if val != pre_val:
            signal = read_signal(val)
            pre_val = val
        if start[idx] + 1 >= end[idx]:
            if region == 'T':
                start[idx] = end[idx] - 13
            elif region == 'P':
                end[idx] = start[idx] + 13
        sig = signal[start[idx]: end[idx]]

        mean = np.mean(sig)
        std_val = np.std(sig, ddof=1)
        n = len(sig)
        kurtosis[idx] = (1 / n) * np.sum((sig - mean) ** 4)
        skewness[idx] = kurtosis[idx] / (std_val ** 3)
    return np.array([kurtosis, skewness])


def get_features(is_training_set=True):
    if is_training_set:
        ante_rr, post_rr = reduce((lambda x, y: np.hstack((x, y))), map(get_rr_interval, label_list))
        p_kur, p_skew = reduce((lambda x, y: np.hstack((x, y))), map(get_p_feature, label_list))
        t_kur, t_skew = reduce((lambda x, y: np.hstack((x, y))), map(get_t_feature, label_list))
        qrs_kur, qrs_skew = reduce((lambda x, y: np.hstack((x, y))), map(get_qrs_feature, label_list))
    return np.vstack((ante_rr, post_rr, p_kur, p_skew, t_kur, t_skew, qrs_kur, qrs_skew)).transpose()


def get_labels(is_training_set):
    if is_training_set:
        return np.hstack((np.full(4000, 0), np.full(4000, 1), np.full(4000, 2), np.full(4000, 3), np.full(4000, 4)))
    else:
        return np.hstack((np.full(1000, 0), np.full(1000, 1), np.full(1000, 2), np.full(1000, 3), np.full(1000, 4)))
