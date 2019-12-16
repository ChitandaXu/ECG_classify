import numpy as np
from ecg_classify.constants import TRAIN_SIZE, TEST_SIZE, CLASS_NUM


class Label:
    def gen(self):
        return


class SingleLabel(Label):
    def gen(self):
        return np.full(20000, 0), np.full(5000, 1)


class MultiLabel(Label):
    def gen(self):
        train_scale = TRAIN_SIZE
        test_scale = TEST_SIZE
        train_labels = np.zeros(train_scale * CLASS_NUM)
        test_labels = np.zeros(test_scale * CLASS_NUM)
        for i in range(CLASS_NUM):
            train_labels[train_scale * i: train_scale * (i + 1)] = i

        for i in range(CLASS_NUM):
            test_labels[test_scale * i: test_scale * (i + 1)] = i

        return train_labels, test_labels

