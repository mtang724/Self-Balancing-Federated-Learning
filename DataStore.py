import numpy as np
import pickle
import struct
import os


class DataStore:
    def __init__(self):
        self.train_feature = np.array([])
        self.train_label = np.array([])
        self.test_feature = np.array([])
        self.test_label = np.array([])

    def get_input(self, name):
        if name == 'cifar':
            dimension_size = 3027
            self.train_feature = np.empty((0, dimension_size))
            for i in range(1, 6):
                with open('./data/cifar/data_batch_{}'.format(i), 'rb') as fo:
                    dic = pickle.load(fo, encoding='bytes')
                self.train_feature = np.vstack((self.train_feature, dic[b'data']))
                self.train_label = np.vstack((self.train_label, dic[b'labels']))

            with open('./data/cifar/test_batch', 'rb') as fo:
                dic = pickle.load(fo, encoding='bytes')
            self.test_feature = dic[b'data']
            self.test_label = dic[b'labels']

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
