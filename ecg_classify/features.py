import numpy as np
from scipy.signal import argrelmin


class Feature:
    def execute(self):
        return


class Skew(Feature):
    def execute(self, sig, lo, hi):
        cur = sig[lo: hi]
        mean = np.mean(cur)
        std_val = np.std(cur, ddof=1)
        n = len(cur)
        skew = 1 / (n - 1) * np.sum((cur - mean) ** 4) / (std_val ** 3)
        return skew


class Kur(Feature):
    def execute(self, sig, lo, hi):
        cur = sig[lo: hi]
        mean = np.mean(cur)
        std_val = np.std(cur, ddof=1)
        n = len(cur)
        skew = 1 / (n - 1) * np.sum((cur - mean) ** 4)
        return skew


class MinDiff(Feature):
    def execute(self, sig, lo, hi):
        mid = lo + (hi - lo) // 2
        left = sig[lo: mid]
        right = sig[mid: hi]
        min_diff = min(right) - min(left)
        return min_diff


class RWidth(Feature):
    def execute(self, sig, lo, hi):
        mid = lo + (hi - lo) // 2
        left = sig[lo: mid]
        right = sig[mid: hi]
        left_idx = np.argmin(left)
        right_idx = np.argmin(right)
        r_width = right_idx - left_idx + len(left)
        return r_width


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
