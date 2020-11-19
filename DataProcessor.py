import numpy as np
import collections
from scipy import special as sp


class DataProcessor:
    def __init__(self):
        pass

    @staticmethod
    def get_size_difference(a, b):
        print("size difference: {} and {}".format(len(a), len(b)))

    # arr shape: (number of cluster, number of data for each cluster)
    def get_local_difference(self, arr):
        n = len(arr)
        res = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                res[i][j] = self.get_kl_divergence(arr[i], arr[j])
        return res

    @staticmethod
    def get_global_difference(arr):
        c1 = collections.Counter(arr)
        return min(c1.values()), max(c1.values())

    @staticmethod
    def get_kl_divergence(input1, input2):
        c1, c2 = collections.Counter(input1), collections.Counter(input2)
        d1, d2 = [], []
        for key in c1.keys():
            d1.append(c1[key] / len(input1))
            d2.append(c2[key] / len(input2))
        return sp.rel_entr(d1, d2)



