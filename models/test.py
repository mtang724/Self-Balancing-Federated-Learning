#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix


def test_img(net_g, datatest, args, is_self_balanced, imbalanced_way):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    y_pred_list = []
    y_true_list = []
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        y_pred_squeeze = torch.squeeze(y_pred).tolist()
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        y_true = target.tolist()
        y_pred_list += y_pred_squeeze
        y_true_list += y_true

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    print(datatest.type)
    print(classification_report(y_true_list, y_pred_list))
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    if args.confusion_matrix and datatest.type == "test":
        cf_matrix = confusion_matrix(y_true_list, y_pred_list)
        plt.figure(figsize=(10, 7))
        sn.heatmap(cf_matrix, annot=True, fmt='g')
        plt.savefig("test_figures/confusion_matrix_" + is_self_balanced + "_" + imbalanced_way + ".png")
    return accuracy, test_loss

