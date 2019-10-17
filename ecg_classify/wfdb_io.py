# -*- coding: utf-8 -*-

"""Main module."""
import wfdb
import numpy as np
from enum import Enum


class DataSetType(Enum):
    TRAINING = 0
    TEST = 1


class NormalBeat:
    beat_type = 'N'
    training_set_dict = {
        101: 1000,
        103: 1000,
        108: 1000,
        112: 1000
    }
    test_set_dict = {
        100: 1000
    }


class LBBBBeat:
    beat_type = 'L'
    training_set_dict = {
        109: 1500,
        111: 1500,
        207: 1000
    }
    test_set_dict = {
        214: 1000
    }


class RBBBBeat:
    beat_type = 'R'
    training_set_dict = {
        118: 1500,
        124: 1500,
        231: 1000
    }
    test_set_dict = {
        212: 1000
    }


class APCBeat:
    beat_type = 'A'
    training_set_dict = {
        101: 3,
        108: 4,
        112: 2,
        118: 96,
        124: 2,
        200: 30,
        207: 106,
        209: 383,
        232: 1374
    }
    test_set_dict = {
        100: 33,
        201: 30,
        202: 36,
        205: 3,
        213: 25,
        220: 94,
        222: 207,
        223: 72
    }


class VPCBeat:
    beat_type = 'V'
    training_set_dict = {
        106: 520,
        116: 109,
        118: 16,
        119: 443,
        124: 47,
        200: 825,
        203: 444,
        208: 991,
        215: 164,
        221: 79,
        228: 362
    }
    test_set_dict = {
        105: 41,
        201: 198,
        205: 71,
        214: 256,
        219: 64,
        223: 370
    }


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
    signal, signal_info = wfdb.rdsamp(file_path, samp_from, samp_to, channels)
    return [signal, signal_info]


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


def duplicate_array(origin_array):
    new_array = np.concatenate([origin_array, origin_array], axis=0)
    return new_array


def generate_sample_by_heartbeat(heartbeat, data_set_type):
    if not isinstance(heartbeat, (NormalBeat, LBBBBeat, RBBBBeat, APCBeat, VPCBeat)):
        raise Exception("Heart beat type invalid.")
    if not isinstance(data_set_type, DataSetType):
        raise Exception("Data type is invalid, please specify 'TRAINING' or 'TEST'.")

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
    r_loc_set = np.empty(data_set.shape[0])
    prev_r_loc_set = np.empty(data_set.shape[0])
    next_r_loc_set = np.empty(data_set.shape[0])

    keys = list(cur_dict.keys())
    for idx, val in enumerate(keys):
        sig = read_signal(val)[0][:, 0]
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
    if heartbeat.beat_type == APCBeat.beat_type:
        # sample number need to be doubled since the number of APC type is about 2000
        data_set = duplicate_array(data_set)
        r_loc_set = duplicate_array(r_loc_set)
        prev_r_loc_set = duplicate_array(prev_r_loc_set)
        next_r_loc_set = duplicate_array(next_r_loc_set)
    return [data_set, r_loc_set, prev_r_loc_set, next_r_loc_set]


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
