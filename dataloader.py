#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
#                    _          _
#                .__(.)<  ??  >(.)__.
#                 \___)        (___/ 
# @Time    : 2022/3/17 下午9:45
# @Author  : wds -->> hellowds2014@gmail.com
# @File    : dataloader.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import scipy.io as sio
from scipy import sparse
import torch


class CustomDataset(Dataset):

    def __init__(self, data_file, dataname='20ng', mode='train'):
        self.mode = mode
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        if dataname == '20ng_20':
            self.data = data['bow']
            self.label = np.squeeze(data['label20'])
            self.voc = data['voc']
            train_id = data['train_id']
            test_id = data['test_id']
            if mode == 'train':
                self.data = self.data[train_id]
                self.label = self.label[train_id]
            elif mode == 'test':
                self.data = self.data[test_id]
                self.label = self.label[test_id]

        if dataname == '20ng_6':
            self.data = data['bow']
            self.label = np.squeeze(data['label6'])
            self.voc = data['voc']
            train_id = data['train_id']
            test_id = data['test_id']
            if mode == 'train':
                self.data = self.data[train_id]
                self.label = self.label[train_id]
            elif mode == 'test':
                self.data = self.data[test_id]
                self.label = self.label[test_id]

        elif dataname == 'reuters':
            self.data = data['bow']
            self.voc = data['voc']
            self.label = np.zeros(self.data.shape[0])

        elif dataname == 'rcv2':
            self.voc = data['voc']
            if mode == 'train':
                self.data = data['train_bow']
                self.label = data['train_label']
            elif mode == 'test':
                self.data = data['test_bow']
                self.label = data['test_label']

        elif dataname == 'web':
            self.voc = data['voc']
            if mode == 'train':
                self.data = data['train_bow'].T
                self.label = data['train_label']
            elif mode == 'test':
                self.data = data['test_bow'].T
                self.label = data['test_label']

        elif dataname == 'tmn':
            self.voc = data['voc']
            if mode == 'train':
                self.data = data['train_bow'].T
                self.label = data['train_label']
            elif mode == 'test':
                self.data = data['test_bow'].T
                self.label = data['test_label']


        elif dataname == 'dp':
            self.voc = data['voc']
            if mode == 'train':
                self.data = data['train_bow'].T
                self.label = data['train_label']
            elif mode == 'test':
                self.data = data['test_bow'].T
                self.label = data['test_label']

    def __getitem__(self, index):
        try:
            bow = np.squeeze(self.data[index].toarray())
        except:
            bow = np.squeeze(self.data[index])
        return bow, np.squeeze(self.label[index])

    def __len__(self):
        return self.data.shape[0]


def dataloader(data_file, dataname='20ng_6', mode='train', batch_size=500, shuffle=True, drop_last=True, num_workers=4):
    dataset = CustomDataset(data_file, dataname=dataname, mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last), dataset.voc

            





