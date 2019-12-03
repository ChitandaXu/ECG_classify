import numpy as np
import os
import pandas as pd
from ecg_classify.constants import FEATURE_NUM, DS1, LABEL_DICT, DS2
from ecg_classify.gen_feature import gen_feature
from ecg_classify.wfdb_io import compute_number


def read_data(force=False):
    if (not (os.path.isfile('train.csv') and os.path.isfile('test.csv'))) or force:
        write_data(True)
        write_data(False)
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    return df_train, df_test


def compute_ds_number(symbol, data_set):
    count = 0
    for num in data_set:
        labels = LABEL_DICT[symbol]
        count += compute_number(num, labels)
    return count


def gen_data(symbol, is_training):
    if is_training:
        data_set = DS1
    else:
        data_set = DS2
    count = compute_ds_number(symbol, data_set)
    labels = LABEL_DICT[symbol]
    res = np.empty((count, FEATURE_NUM), dtype='<U32')
    idx = 0
    for num in data_set:
        for label in labels:
            ft = gen_feature(num)
            cur = ft[ft[:, -1] == label]
            length = cur.shape[0]
            res[idx: idx + length] = cur
            idx = idx + length
    return res


def gen_label(is_training):
    if is_training:
        data_set = DS1
    else:
        data_set = DS2
    n_count = compute_ds_number('N', data_set)
    s_count = compute_ds_number('S', data_set)
    v_count = compute_ds_number('V', data_set)
    f_count = compute_ds_number('F', data_set)
    q_count = compute_ds_number('Q', data_set)
    return np.hstack((np.full(n_count, 0), np.full(s_count, 1),
                      np.full(v_count, 2), np.full(f_count, 3), np.full(q_count, 4)))


def write_data(is_training=True):
    n_type = gen_data('N', is_training)
    s_type = gen_data('S', is_training)
    v_type = gen_data('V', is_training)
    f_type = gen_data('F', is_training)
    q_type = gen_data('Q', is_training)
    res = np.vstack((n_type, s_type, v_type, f_type, q_type))
    df = pd.DataFrame(res)
    if is_training:
        df.to_csv("train.csv", index=False)
    else:
        df.to_csv("test.csv", index=False)
