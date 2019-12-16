import numpy as np
import os
import pandas as pd
from ecg_classify.constants import DIM, heartbeat_factory, CLASS_NUM, TRAIN_SIZE, TEST_SIZE, LABEL_LIST
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
    if symbol == 'A' or (symbol == '/' and is_training):
        half = res.shape[0] // 2
        res = res[0: half]
        res = np.concatenate([res, res])
    return res


def gen_label(is_training_set=True):
    if is_training_set:
        scale = TRAIN_SIZE
    else:
        scale = TEST_SIZE
    labels = np.zeros(scale * CLASS_NUM)
    for i in range(CLASS_NUM):
        labels[scale * i: scale * (i + 1)] = i
    return labels


def __write_data(is_training=True):
    if is_training:
        scale = TRAIN_SIZE
    else:
        scale = TEST_SIZE
    res = np.empty((scale * CLASS_NUM, DIM), dtype='<U32')
    for i in range(CLASS_NUM):
        res[scale * i: scale * (i + 1)] = gen_data(LABEL_LIST[i], is_training)
    df = pd.DataFrame(res)
    if is_training:
        df.to_csv("train.csv", index=False)
    else:
        df.to_csv("test.csv", index=False)
