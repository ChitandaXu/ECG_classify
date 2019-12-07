# -*- coding: utf-8 -*-

"""Main module."""
import wfdb
import numpy as np


def read_sig(data_number, samp_from=0, samp_to=None, channels=[0, 1], dir_path='../data/mit-bih'):
    file_path = dir_path + "/" + str(data_number)
    signal = wfdb.rdsamp(file_path, samp_from, samp_to, channels)[0][:, 0]
    return signal


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
