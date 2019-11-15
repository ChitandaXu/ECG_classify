import numpy as np
import os
import pandas as pd
from ecg_classify.constants import heartbeat_factory, FEATURE_NUM
from ecg_classify.gen_feature import gen_feature


def read_data(rescale=True):
    if not (os.path.isfile('train.csv') and os.path.isfile('test.csv')):
        __write_data(True, rescale)
        __write_data(False, rescale)
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    return df_train, df_test


def gen_data(symbol, is_training=True, rescale=True):
    heartbeat = heartbeat_factory(symbol)
    if is_training:
        num_dict = heartbeat.training_set_dict
        num_list = list(num_dict.keys())
        res = np.empty((4000, FEATURE_NUM), dtype='<U32')
    else:
        num_dict = heartbeat.test_set_dict
        num_list = list(num_dict.keys())
        res = np.empty((1000, FEATURE_NUM), dtype='<U32')
    cur = 0
    for num in num_list:
        feature = gen_feature(num, rescale)
        val = num_dict[num]
        res[cur: cur + val] = feature[feature[:, -1] == heartbeat.beat_type][0: val]
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


def __write_data(is_training=True, rescale=True):
    if is_training:
        res = np.empty((20000, FEATURE_NUM), dtype='<U32')
        scale = int(20000 / 5)
    else:
        res = np.empty((5000, FEATURE_NUM), dtype='<U32')
        scale = int(5000 / 5)
    res[0: scale] = gen_data('N', is_training, rescale)
    res[scale: 2 * scale] = gen_data('L', is_training, rescale)
    res[2 * scale: 3 * scale] = gen_data('R', is_training, rescale)
    res[3 * scale: 4 * scale] = gen_data('A', is_training, rescale)
    res[4 * scale: 5 * scale] = gen_data('V', is_training, rescale)
    df = pd.DataFrame(res)
    if is_training:
        df.to_csv("train.csv", index=False)
    else:
        df.to_csv("test.csv", index=False)
