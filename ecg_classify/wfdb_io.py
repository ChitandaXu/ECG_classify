# -*- coding: utf-8 -*-

"""Main module."""
import wfdb
import pywt
import numpy as np
from ecg_classify.constants import NormalBeat, LBBBBeat, RBBBBeat, APCBeat, VPCBeat, DataSetType


def read_signal(data_number, samp_from=0, samp_to=None, channels=[0, 1], dir_path='../data/mit-bih'):
    """
    Read a signal from data file

    :param data_number: data number of the ECG signal data
    :param samp_from: where signal start
    :param samp_to: where signal end
    :param channels: channels
    :param dir_path: folder path of the data file
    :return: signal data, signal information
    """
    file_path = dir_path + "/" + str(data_number)
    signal = wfdb.rdsamp(file_path, samp_from, samp_to, channels)[0][:, 0]
    return signal


def read_annotation(data_number, samp_from=0, samp_to=None, extension='atr', return_label=['symbol'],
                    dir_path='../data/mit-bih'):
    """
    read annotation from data file

    :param data_number: data number of the ECG signal data
    :param samp_from: where signal start
    :param samp_to: where signal end
    :param extension: file extension, eg: atr
    :param return_label: at lease one of them: [‘label_store’, ‘symbol’, ‘description’, ‘n_occurrences’]
    :param dir_path: folder path of the data file
    :return: annotation
    """
    file_path = dir_path + "/" + str(data_number)
    ann = wfdb.rdann(file_path, extension, samp_from, samp_to, False, None, return_label)
    return ann


def read_all_signals(dir_path='../data/mit-bih'):
    """
    Read all signal from MIT-BIH database
    102, 104 do not contain MLII, should be excluded
    114 MLII is in channel 2

    :param dir_path: dir path
    :return: signals
    """
    arr = np.loadtxt(dir_path + "/RECORDS").astype('int32')
    number_of_signal = arr.shape[0]
    len_of_signal = 650000

    # 102, 104 should be excluded
    signals = np.empty((len_of_signal, number_of_signal - 2))
    index = 0
    for i in range(number_of_signal):
        signal = read_signal(arr[i], dir_path=dir_path)[0]
        if arr[i] == 102 or arr[i] == 104:
            continue
        elif arr[i] == 114:
            signals[:, index] = signal[:, 1]
        else:
            signals[:, index] = signal[:, 0]
        index = index + 1
    return signals


def duplicate_array(*args):
    res = []
    for i in range(len(args)):
        res.append(np.concatenate([args[i], args[i]], axis=0))
    return res


def heartbeat_factory(heartbeat_symbol):
    if heartbeat_symbol == 'N':
        return NormalBeat()
    elif heartbeat_symbol == 'L':
        return LBBBBeat()
    elif heartbeat_symbol == 'R':
        return RBBBBeat()
    elif heartbeat_symbol == 'A':
        return APCBeat()
    elif heartbeat_symbol == 'V':
        return VPCBeat()
    else:
        raise Exception('Invalid heartbeat type')


def init_set(heartbeat, data_set_type=DataSetType.TRAINING):
    if data_set_type == DataSetType.TRAINING:
        if heartbeat.beat_type == APCBeat.beat_type:
            data_set = np.empty((2000, 300))
        else:
            data_set = np.empty((4000, 300))
        cur_dict = heartbeat.training_set_dict
    else:
        if heartbeat.beat_type == APCBeat.beat_type:
            data_set = np.empty((500, 300))
        else:
            data_set = np.empty((1000, 300))
        cur_dict = heartbeat.test_set_dict
    data_size = data_set.shape[0]
    template = np.empty(data_size).astype(int)
    return [cur_dict, data_set, template.copy(), template.copy(), template.copy(), template.copy()]


def generate_sample_by_heartbeat(heartbeat_symbol, data_set_type, need_denoise=True):
    """
    generate sample for training by specify heartbeat type

    :param heartbeat_symbol: HeartBeat Symbol, eg: 'N', 'L', 'R', 'A', 'V'
    :param data_set_type: data set type, 'Training' or 'Test'
    :param need_denoise: choose whether to denoise, default is True
    :return: sample for training
    """
    if heartbeat_symbol not in ['N', 'L', 'R', 'A', 'V']:
        raise Exception("Invalid heartbeat type")
    if not isinstance(data_set_type, DataSetType):
        raise Exception("Data type is invalid, please specify 'TRAINING' or 'TEST'.")
    heartbeat = heartbeat_factory(heartbeat_symbol)
    [cur_dict, data_set, r_loc_set, prev_r_loc_set, next_r_loc_set, number_set] = init_set(heartbeat, data_set_type)

    keys = list(cur_dict.keys())
    for idx, val in enumerate(keys):
        sig = read_signal(val)
        if need_denoise:
            sig = denoise(sig)
        ann = read_annotation(val)
        origin_r_loc = ann.sample
        origin_symbol = np.asarray(ann.symbol)
        r_idx = np.where(origin_symbol == heartbeat.beat_type)[0]
        r_idx = r_idx[np.logical_and(r_idx >= 2, r_idx < len(origin_r_loc) - 1)]
        prev_r_loc = origin_r_loc[r_idx - 1]
        next_r_loc = origin_r_loc[r_idx + 1]
        cur_r_loc = origin_r_loc[r_idx]
        heartbeat_samples = get_heartbeat_samples(cur_r_loc, sig)
        if idx == 0:
            start = 0
        else:
            start = end
        cur_count = cur_dict[keys[idx]]
        end = cur_count + start
        data_set[start: end] = heartbeat_samples[0: cur_count]
        r_loc_set[start: end] = cur_r_loc[0: cur_count]
        prev_r_loc_set[start: end] = prev_r_loc[0: cur_count]
        next_r_loc_set[start: end] = next_r_loc[0: cur_count]
        number_set[start: end] = keys[idx]

    if heartbeat.beat_type == APCBeat.beat_type:
        # sample number need to be doubled since the number of APC type is about 2000
        return duplicate_array(data_set, r_loc_set, prev_r_loc_set, next_r_loc_set, number_set)
    return [data_set, r_loc_set, prev_r_loc_set, next_r_loc_set, number_set]


def get_heartbeat_samples(r_peak_idx, signal):
    num_of_r_wave = len(r_peak_idx)
    beats = np.empty((num_of_r_wave, 300))
    for i in range(num_of_r_wave):
        start = r_peak_idx[i] - 150
        end = r_peak_idx[i] + 150
        if start < 0 or end > 650000:
            raise Exception('start point or end point is out of range.')
        beats[i] = signal[start: end]
    return beats


def get_sig_type(dir_path='../data/mit-bih'):
    arr = np.loadtxt(dir_path + "/RECORDS").astype('int32')
    number_of_signal = arr.shape[0]
    sig_type = np.empty((number_of_signal, 2), dtype='<U4')
    for i in range(number_of_signal):
        sig_type[i] = read_signal(arr[i], 0, 50)[1]['sig_name']
    return sig_type


def denoise(signal):
    """
    Denoise by db6 with level 6

    :param signal: ECG signal
    :return: signal after denoise
    """
    coeffs = pywt.wavedec(signal, 'db6', level=6)
    cA6, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # only remain cD6 -> cD3
    cA6 = np.zeros(cA6.shape)
    cD2 = np.zeros(cD2.shape)
    cD1 = np.zeros(cD1.shape)
    coeffs = [cA6, cD6, cD5, cD4, cD3, cD2, cD1]
    return pywt.waverec(coeffs, 'db6')
