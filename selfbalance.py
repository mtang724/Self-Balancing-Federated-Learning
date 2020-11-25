#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import math


def zscore(N, labels, K, datasets):
    """
    The FL Server part (Algorithm 2)
    The td and ta are the downsampling threshold and augmentation threshold
    Set Ta = -1/Td , the recommended value of td is 3.0 or 3.5
    The Rad is the ratio we use to control how many augmentations are generated or how many samples are retained.
    N : Number of classes
    K : Total number of clients
    labels : All the data label 
    Ydown : set of majority class
    Yaug : set of minority class
    datasets : K clients datasets
    """
    # 2 : Initialize
    Td = 3
    Ta = -1/Td
    Rad = np.zeros(N)
    # 3 : Calculate the data size of each class C
    C = np.zeros(N)
    for i in labels:
        C[i] = C[i]+1
    # 4 : Calculate the mean m and the standard deviation s of C
    mean = np.mean(C)
    std = np.std(C, ddof = 1)
    # 5 : Calculate the z-score
    z = (C-mean)/std
    # 6-12 :
    Ydown = set()
    Yaug = set()
    for y in range(0, N):
        if z[y] < Ta:
            Yaug.add(y)
            Rad[y] = (std * math.sqrt(abs(z[y]/Ta)) + mean )/C[y]
        elif z[y] > Td:
            Ydown.add(y)
            Rad[y] = (std * math.sqrt(z[y]*Td) + mean )/C[y]
    # 13 : Send Yaug, Ydown, Rad to all clients =================================================== ???
    """
    The Clients part (Algorithm 2)
    """
    # 15-22 :
    for k in range(0, K):
        for (x,y) in datasets[k]: # 具体改一下
            if y in Ydown:
                Downsample(x, y, Rad[y])
            elif y in Yaug:
                Augment(x, y, Rad[y]-1)
        datasets[k].add() # 加 augment k 入 client k 的数据集
        ShuffleDataset(k) # 随机数据集，感觉没必要，可以提取的时候用随机idx

def Downsample(): # 加上

def Augment(): # 加上


if __name__ == '__main__':
    print('self balance functions')
