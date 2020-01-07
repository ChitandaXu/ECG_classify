import numpy as np
from scipy.signal import argrelmin


class Feature:
    def execute(self):
        return


class Skew(Feature):
    def execute(self, sig, lo, hi):
        cur = sig[lo: hi]
        mean = np.mean(cur)
        std_val = np.std(cur)
        if std_val == 0:
            std_val = 1e-6
        n = len(cur)
        skew = 1 / n * np.sum((cur - mean) ** 4) / (std_val ** 3)
        return skew


class Kur(Feature):
    def execute(self, sig, lo, hi):
        cur = sig[lo: hi]
        mean = np.mean(cur)
        n = len(cur)
        skew = 1 / n * np.sum((cur - mean) ** 4)
        return skew


class MinDiff(Feature):
    def execute(self, sig, lo, hi):
        cur = sig[lo: hi]
        idx = np.argmin(cur)
        if abs(idx) <= 3:
            sig = -sig
        mid = lo + (hi - lo) // 2
        left = sig[lo: mid]
        right = sig[mid: hi]
        min_diff = min(right) - min(left)
        return min_diff


class QrsWidth(Feature):
    def execute(self, sig, lo, hi):
        mid = lo + (hi - lo) // 2
        left = sig[lo: mid]
        right = sig[mid: hi]
        left_idx = np.argmin(left)
        right_idx = np.argmin(right)
        qrs_width = right_idx - left_idx + len(left)
        return qrs_width


class Slope(Feature):
    def execute(self, sig, lo, hi):
        cur = sig[lo: hi]
        min_arr = argrelmin(cur)[0]
        if min_arr.size == 0:
            idx = np.argmin(cur)
            if idx == 0:
                idx = np.argmax(cur)
        else:
            idx = min_arr[0]
            if (cur[0] - cur[idx]) < 1.5 and min_arr.size > 1:
                idx = min_arr[1]
        slope = np.abs((cur[0] - cur[idx]) / idx)
        return slope


class PRInterval(Feature):
    def execute(self, sig, lo, hi):
        if lo < 0:
            lo = 0
        sig = revert_sig(sig, lo, hi)
        p_region = sig[lo: hi - 20]
        p_idx = np.argmax(p_region)
        return hi - p_idx


def revert_sig(sig, lo, hi):
    cur = sig[lo: hi]
    idx = np.argmin(cur)
    if abs(idx) <= 3:
        sig = -sig
    return sig
