import numpy as np
import os
import pandas as pd
from ecg_classify.constants import DIM, CLASS_NUM, LABEL_LIST, DS1, DS2
from ecg_classify.dataset import DataSet
from ecg_classify.gen_feature import gen_feature


def read_data(refresh=False):
    ds1 = DataSet(DS1)
    ds2 = DataSet(DS2)
    if (not (os.path.isfile('train.csv') and os.path.isfile('test.csv'))) or refresh:
        write_data(True, ds1)
        write_data(False, ds2)
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    return df_train, df_test


def gen_data(label, ds):
    loc = LABEL_LIST.index(label)
    ds_nums = ds.get_ds_nums()
    ds_num = ds_nums[loc]  # 获取该数据集该类别的样本个数

    res = np.empty((ds_num, DIM), dtype='<U32')
    idx = 0
    for num in ds.get_dataset():
        ft = gen_feature(num)
        cur = ft[ft[:, -1] == label]
        length = cur.shape[0]
        res[idx: idx + length] = cur
        idx = idx + length
    return res


def write_data(is_training, ds):
    size = ds.get_size()
    ds_nums = ds.get_ds_nums()
    res = np.empty((size, DIM), dtype='<U32')
    lo = 0
    hi = 0
    for i in range(CLASS_NUM):
        hi += ds_nums[i]
        res[lo: hi] = gen_data(LABEL_LIST[i], ds)
        lo = hi
    df = pd.DataFrame(res)
    if is_training:
        df.to_csv("train.csv", index=False)
    else:
        df.to_csv("test.csv", index=False)
