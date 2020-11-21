import numpy as np
import collections
from scipy import special as sp
import random


class DataSeparator:
    def __init__(self):
        pass

    def seperate_dataset(self, feature, label, device, num, alpha):
        f, l = [[] for i in range(10)], [[] for i in range(10)]
        poll_feature, poll_label = [], [] # feature and label in the poll
        
        # seperate dataset by label 0 - 9
        c = 0
        for i in range(len(label)): 
            f[label[i]].append(feature[i])
            l[label[i]].append(label[i])
            c = c + 1
        
        #  Divide the data set into groups which equal to the number of devices
        res_f = [[] for i in range(device)]
        res_l = [[] for i in range(device)]
        for i in range(device):
            cur = i % 10
            res_f[i].extend(f[cur][:num])
            del (f[cur][num])
            res_l[i].extend(l[cur][:num])
            del (l[cur][:num])

        # extract num*alpha items from each group into the sample poll, then shuffle the poll
        count = int(num * alpha)
        for j in range(device):
            poll_feature.extend(res_f[j][(len(res_f[j]) - count):len(res_f[j])])
            del (res_f[j][(len(res_f[j]) - count):len(res_f[j])])
            poll_label.extend(res_l[j][(len(res_l[j]) - count):len(res_l[j])])
            del (res_l[j][(len(res_l[j]) - count):len(res_l[j])])
        random.seed(0)
        random.shuffle(poll_feature)
        random.seed(0)
        random.shuffle(poll_label)
        
        # extract num*alpha items from sample poll then put back to each group
        for i in range(device):
            res_f[i].extend(poll_feature[:count])
            del (poll_feature[:count])
            res_l[i].extend(poll_label[:count])
            del (poll_label[:count])
        return res_f, res_l 