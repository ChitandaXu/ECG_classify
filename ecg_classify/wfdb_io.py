# -*- coding: utf-8 -*-

"""Main module."""
import wfdb
import numpy as np

from ecg_classify.constants import NUM_TUPLE, LABEL_DICT, SYMBOL_TUPLE


def read_sig(data_number, samp_from=0, samp_to=None, channels=[0, 1], dir_path='../data/mit-bih'):
    signal = __read_raw(data_number, samp_from, samp_to, channels, dir_path)[0][:, 0]
    if data_number == 114:
        signal = __read_raw(data_number, samp_from, samp_to, channels, dir_path)[0][:, 1]
    return signal


def print_channel(data_number):
    print(__read_raw(data_number)[1]['sig_name'])


def print_all_channels():
    with open('../data/mit-bih/RECORDS', 'r') as f:
        res = f.read()
    num_list = res.split('\n')
    for num in num_list:
        if num == '':
            continue
        print(num)
        print_channel(num)


def __read_raw(data_number, samp_from=0, samp_to=None, channels=[0, 1], dir_path='../data/mit-bih'):
    file_path = dir_path + "/" + str(data_number)
    return wfdb.rdsamp(file_path, samp_from, samp_to, channels)


def read_symbol(data_number, sampfrom=0, sampto=None):
    return np.array(__read_ann(data_number, sampfrom, sampto).symbol)


def read_sample(data_number, sampfrom=0, sampto=None):
    return __read_ann(data_number, sampfrom, sampto).sample


def __read_ann(data_number, sampfrom=0, sampto=None, dir_path='../data/mit-bih'):
    file_path = dir_path + "/" + str(data_number)
    ann = wfdb.rdann(file_path, extension='atr', sampfrom=sampfrom, sampto=sampto)
    return ann


def compute_number(num, labels):
    symbols = read_symbol(num)[2: -2]  # 3nd -> (n-2)th
    res = 0
    for label in labels:
        res += len(symbols[symbols == label])
    return res


def compute_total_number():
    for num in NUM_TUPLE:
        for symbol in SYMBOL_TUPLE:
            labels = LABEL_DICT[symbol]
            count = compute_number(num, labels)
            print(num, symbol, count)
