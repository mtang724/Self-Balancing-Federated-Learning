import os
import pickle
import struct
import collections
import numpy as np
import torch
from torchvision import transforms
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
        self.local_train_index = None

        self.size_class = None
        self.size_device = None
        self.size_feature = None

        self.transform = None

        self.type = 'train'
        self.data_source = None

    def __len__(self):
        return len(self.global_train_label)

    def __getitem__(self, idx):
        if self.type == 'train':
            feature, label = self.train_feature[idx], self.train_label[idx]
        else:
            feature, label = self.test_feature[idx], self.test_label[idx]
        if self.data_source == "cifar":
            feature = feature.reshape(32, 32, 3).astype(np.float32)
        elif self.data_source == "mnist":
            feature = feature.reshape(28, 28, 1).astype(np.float32)
        img = self.transform(feature)
        return img, label

    # region data storage
    def get_input(self, name):
        self.__init__()
        self.data_source = name
        if name == 'cifar':
            dimension_size = 3072
            self.train_feature = np.empty((0, dimension_size), dtype=np.int)
            self.train_label = np.array([], dtype=np.int)
            for i in range(1, 6):
                with open('./data/cifar/data_batch_{}'.format(i), 'rb') as fo:
                    dic = pickle.load(fo, encoding='bytes')
                self.train_feature = np.vstack((self.train_feature, dic[b'data']))
                self.train_label = np.hstack((self.train_label, np.array(dic[b'labels'], dtype=np.int)))
            self.train_feature = self.train_feature.reshape(len(self.train_feature), 3, 32, 32).transpose(0, 2, 3, 1)
            self.train_feature = self.train_feature.reshape(len(self.train_feature), -1)
            with open('./data/cifar/test_batch', 'rb') as fo:
                dic = pickle.load(fo, encoding='bytes')
            self.test_feature = dic[b'data']
            self.test_label = np.array(dic[b'labels'], dtype=np.int)
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

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
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        self.size_class = len(set(self.train_label) | set(self.test_label))
        self.size_feature = self.train_feature.shape[1]

        self.train_feature = self.train_feature.astype(int)
        self.train_label = self.train_label.astype(int)
        self.test_feature = self.test_feature.astype(int)
        self.test_label = self.test_label.astype(int)

    # endregion

    # region imbalance evaluation
    @staticmethod
    def get_size_difference(arr):
        for i, a in enumerate(arr):
            print('the {}th device size: {}'.format(i, len(a)))

    # arr shape: (number of cluster, number of data for each cluster)
    def get_local_difference(self, arr):
        n = len(arr)
        res = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                res[i][j] = self.get_kl_divergence(arr[i], arr[j])
        return res

    def get_global_difference(self, arr):
        c1 = collections.Counter(arr).values()
        return [0] * (self.size_class - len(c1)) + sorted(c1)

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
        # num_device indicates the number of the devices
        # device_size is the size of each device(it's an integer, not a list, as each device should have the same size)
        # alpha is the ratio to control KL.
        # Alpha = 0 means full random sampling, no imbalance
        # Alpha = 1 means no sampling, each device takes only one class
        self.size_device = num_device
        self.local_train_feature = []
        self.local_train_label = []

        # separate data set by label 0 - 9
        feature_by_class = []
        for i in range(self.size_class):
            need_idx = np.where(self.train_label == i)[0]
            feature_by_class.append(self.train_feature[need_idx])

        remain_size = int(device_size * alpha)
        sample_size = device_size - remain_size

        sample_feature_pool = np.array([], dtype=np.int)
        sample_label_pool = np.array([], dtype=np.int)

        # keep the proportion of alpha of the original data in the specific class
        for i in range(self.size_class):
            need_idx = np.arange(len(feature_by_class[i]))
            np.random.shuffle(need_idx)
            step = -1
            for j in range(i, self.size_device, self.size_class):
                step += 1
                select_idx = need_idx[step*remain_size:(step+1)*remain_size]
                self.local_train_feature.append(feature_by_class[i][select_idx])
                self.local_train_label.append(np.repeat(i, remain_size))

            # put the data that not selected into the sample pool
            select_idx = need_idx[(step + 1) * remain_size:]
            if sample_feature_pool.size:
                sample_feature_pool = np.vstack([sample_feature_pool, feature_by_class[i][select_idx]])
            else:
                sample_feature_pool = feature_by_class[i][select_idx]
            sample_label_pool = np.hstack([sample_label_pool, np.repeat(i, len(select_idx))])

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
            self.local_train_label[i] = np.hstack([self.local_train_label[i], sample_label_pool[select_idx]])
        self.refresh_global_data()

    def gen_size_imbalance(self, list_size):
        # generate size imbalance
        # list_size is a size indicates the size of each device
        self.size_device = len(list_size)
        self.local_train_feature = []
        self.local_train_label = []

        need_idx = np.arange(len(self.train_feature))
        np.random.shuffle(need_idx)
        cur_idx = 0
        for s in list_size:
            self.local_train_feature.append(self.train_feature[need_idx[cur_idx:cur_idx+s]])
            self.local_train_label.append(self.train_label[need_idx[cur_idx:cur_idx+s]])
            cur_idx += s
        self.refresh_global_data()

    def gen_global_imbalance(self, num_device, device_size, num_each_class):
        # generate global imbalance
        # num_device indicates the number of the devices
        # device_size is the size of each device(it's an integer, not a list, as each device should have the same size)
        # num_each_class is a list indicating the number for each class
        self.size_device = num_device
        self.local_train_feature = []
        self.local_train_label = []

        feature_by_class = []
        for i in range(self.size_class):
            need_idx = np.where(self.train_label == i)[0]
            feature_by_class.append(self.train_feature[need_idx])
        sample_feature_pool = np.array([], dtype=np.int)
        sample_label_pool = np.array([], dtype=np.int)

        for i in range(self.size_class):
            need_idx = np.arange(len(feature_by_class[i]))
            np.random.shuffle(need_idx)
            if sample_feature_pool.size:
                sample_feature_pool = np.vstack([sample_feature_pool,
                                                 feature_by_class[i][need_idx[:num_each_class[i]]]])
            else:
                sample_feature_pool = feature_by_class[i][need_idx[:num_each_class[i]]]
            sample_label_pool = np.hstack([sample_label_pool, np.repeat(i, num_each_class[i])])

        need_idx = np.arange(len(sample_feature_pool))
        np.random.shuffle(need_idx)
        step = -1
        for i in range(self.size_device):
            step += 1
            select_idx = need_idx[step*device_size:(step+1)*device_size]
            self.local_train_feature.append(sample_feature_pool[select_idx])
            self.local_train_label.append(sample_label_pool[select_idx])
        self.refresh_global_data()

    def refresh_global_data(self):
        # initialize global train features and labels
        self.global_train_feature = np.empty((0, self.size_feature), dtype=np.int)
        self.global_train_label = np.array([], dtype=np.int)
        idx_start = 0
        for i in range(self.size_device):
            self.global_train_feature = np.vstack([self.global_train_feature, self.local_train_feature[i]])
            self.global_train_label = np.hstack([self.global_train_label, self.local_train_label[i]])
            self.local_train_index.append(np.arange(idx_start, idx_start+len(self.local_train_label[i])))
    #  endregion
