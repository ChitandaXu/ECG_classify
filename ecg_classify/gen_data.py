import numpy as np
import os
import pandas as pd
from ecg_classify.constants import DIM, heartbeat_factory
from ecg_classify.gen_feature import gen_feature


def read_data(force=False):
    if (not (os.path.isfile('train.csv') and os.path.isfile('test.csv'))) or force:
        __write_data(True)
        __write_data(False)
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    return df_train, df_test


def gen_data(symbol, is_training=True):
    heartbeat = heartbeat_factory(symbol, is_training)
    if is_training:
        num_list = list(heartbeat.keys())
        res = np.empty((4000, DIM), dtype='<U32')
    else:
        num_list = list(heartbeat.keys())
        res = np.empty((1000, DIM), dtype='<U32')
    cur = 0
    for num in num_list:
        feature = gen_feature(num)
        val = heartbeat[num]
        res[cur: cur + val] = feature[feature[:, -1] == symbol][0: val]
        cur = cur + val
    if symbol == 'A':
        half = int(res.shape[0] / 2)
        res = res[0: half]
        res = np.concatenate([res, res])
    return res


def gen_label(is_training_set):
    if is_training_set:
        return np.hstack((np.full(4000, 0), np.full(4000, 1), np.full(4000, 2), np.full(4000, 3), np.full(4000, 4)))
    else:
        return np.hstack((np.full(1000, 0), np.full(1000, 1), np.full(1000, 2), np.full(1000, 3), np.full(1000, 4)))


def __write_data(is_training=True):
    if is_training:
        res = np.empty((20000, DIM), dtype='<U32')
        scale = int(20000 / 5)
    else:
        res = np.empty((5000, DIM), dtype='<U32')
        scale = int(5000 / 5)
    res[0: scale] = gen_data('N', is_training)
    res[scale: 2 * scale] = gen_data('L', is_training)
    res[2 * scale: 3 * scale] = gen_data('R', is_training)
    res[3 * scale: 4 * scale] = gen_data('A', is_training)
    res[4 * scale: 5 * scale] = gen_data('V', is_training)
    df = pd.DataFrame(res)
    if is_training:
        df.to_csv("train.csv", index=False)
    else:
        df.to_csv("test.csv", index=False)
