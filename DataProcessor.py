import os
import pickle
import struct
import collections
import numpy as np
from scipy import special as sp


class DataProcessor:
    def __init__(self):
        # data load from data set
        self.train_feature = None
        self.train_label = None
        self.test_feature = None
        self.test_label = None
        # data used for train and test totally
        self.global_train_feature = None
        self.global_train_label = None
        # data used for each device
        self.local_train_feature = None
        self.local_train_label = None

        self.size_class = None
        self.size_device = None
        self.size_feature = None

        self.data_source = None

    # region data storage
    def get_input(self, name):
        self.data_source = name
        if name == 'cifar':
            dimension_size = 3072
            self.train_feature = np.empty((0, dimension_size))
            for i in range(1, 6):
                with open('./data/cifar/data_batch_{}'.format(i), 'rb') as fo:
                    dic = pickle.load(fo, encoding='bytes')
                self.train_feature = np.vstack((self.train_feature, dic[b'data']))
                self.train_label = np.hstack((self.train_label, np.array(dic[b'labels'])))
            self.train_feature = self.train_feature.reshape(len(self.train_feature), 3, 32, 32).transpose(0, 2, 3, 1)
            self.train_feature = self.train_feature.reshape(len(self.train_feature), -1)
            with open('./data/cifar/test_batch', 'rb') as fo:
                dic = pickle.load(fo, encoding='bytes')
            self.test_feature = dic[b'data']
            self.test_label = np.array(dic[b'labels'])

        elif name == 'mnist':
            def load_mnist(path, kind='train'):
                labels_path = os.path.join(path, '{}-labels-idx1-ubyte'.format(kind))
                images_path = os.path.join(path, '{}-images-idx3-ubyte'.format(kind))
                with open(labels_path, 'rb') as lbpath:
                    magic, n = struct.unpack('>II', lbpath.read(8))
                    labels = np.fromfile(lbpath, dtype=np.uint8)

                with open(images_path, 'rb') as imgpath:
                    magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
                    images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

                return images, labels

            self.train_feature, self.train_label = load_mnist('./data/mnist', 'train')
            self.test_feature, self.test_label = load_mnist('./data/mnist', 't10k')
        self.size_class = len(set(self.train_label) | set(self.test_label))
        self.size_feature = self.train_feature.shape[1]
    # endregion

    # region imbalance evaluation
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
        c1 = collections.Counter(arr).values()
        return sorted(c1)

    @staticmethod
    def get_kl_divergence(input1, input2):
        c1, c2 = collections.Counter(input1), collections.Counter(input2)
        d1, d2 = [], []
        for key in c1.keys():
            d1.append(c1[key] / len(input1))
            d2.append(c2[key] / len(input2))
        return sum(sp.rel_entr(d1, d2))
    # endregion

    # region generate imbalance data set
    def gen_local_imbalance(self, num_device, device_size, alpha):
        self.size_device = num_device

        # separate data set by label 0 - 9
        feature_by_class = []
        for i in range(self.size_class):
            need_idx = np.where(self.train_label == i)[0]
            feature_by_class.append(self.train_feature[need_idx])

        self.local_train_feature = []
        self.local_train_label = []

        remain_size = int(device_size * alpha)
        sample_size = device_size - remain_size

        sample_feature_pool = np.array([])
        sample_label_pool = np.array([])

        # keep the proportion of alpha of the original data in the specific class
        for i in range(self.size_class):
            need_idx = np.arange(len(feature_by_class[i]))
            np.random.shuffle(need_idx)
            step = -1
            for j in range(i, self.size_device, self.size_class):
                step += 1
                select_idx = need_idx[step*remain_size:(step+1)*remain_size]
                self.local_train_feature.append(feature_by_class[i][select_idx])
                self.local_train_label.append([i for _ in range(remain_size)])

            # put the data that not selected into the sample pool
            select_idx = need_idx[(step + 1) * remain_size:]
            if sample_feature_pool.size:
                sample_feature_pool = np.vstack([sample_feature_pool, feature_by_class[i][select_idx]])
            else:
                sample_feature_pool = feature_by_class[i][select_idx]
            sample_label_pool = np.hstack([sample_label_pool, [i for _ in range(len(select_idx))]])

        # add the data from the sample pool to each device
        need_idx = np.arange(len(sample_feature_pool))
        np.random.shuffle(need_idx)
        step = -1
        for i in range(self.size_device):
            step += 1
            select_idx = need_idx[step*sample_size:(step+1)*sample_size]
            if self.local_train_feature[i].size:
                self.local_train_feature[i] = np.vstack([self.local_train_feature[i], sample_feature_pool[select_idx]])
            else:
                self.local_train_feature[i] = sample_feature_pool[select_idx]
            self.local_train_label[i] += sample_label_pool[select_idx].tolist()

        # initialize global train features and labels
        self.global_train_feature = []
        self.global_train_label = []
        for i in range(self.size_device):
            self.global_train_feature.append(self.local_train_feature[i])
            self.global_train_label += self.local_train_label[i]

        self.global_train_feature = np.array(self.global_train_feature)
        self.global_train_label = np.array(self.global_train_label)

    def gen_size_imbalance(self):
        # achieve size imbalance and give value to
        # self.global_train_feature, self.global_train_label, self.local_train_feature, self.local_train_label
        pass

    def gen_global_imbalance(self):
        # achieve global imbalance and give value to
        # self.global_train_feature, self.global_train_label, self.local_train_feature, self.local_train_label
        pass

    #  endregion
