import numpy as np

from ecg_classify.constants import LABEL_LIST
from ecg_classify.utils import count_num_by_ds


class DataSet:

    def __init__(self, dataset):
        self.ds_nums = np.zeros(6, dtype=int)
        self.dataset = dataset
        for i in range(len(LABEL_LIST)):
            self.ds_nums[i] = count_num_by_ds(dataset, LABEL_LIST[i])

    def get_size(self):
        return np.sum(self.ds_nums)

    def get_ds_nums(self):
        return self.ds_nums

    def get_dataset(self):
        return self.dataset
