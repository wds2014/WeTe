#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
#                    _          _
#                .__(.)<  ??  >(.)__.
#                 \___)        (___/ 
# @Time    : 2022/3/16 下午10:11
# @Author  : wds -->> hellowds2014@gmail.com
# @File    : main.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from dataloader import dataloader
from model import WeTe
from Trainer import Trainer



parser = argparse.ArgumentParser(description='Representing mixtures of word embeddings with mixtures of topic embeddings (WeTe)')
# WeTe options
parser.add_argument('--embedding_dim', type=int, default=100, metavar='N', help='embeddings dimension (default: 100)')
parser.add_argument('--K', type=int, default=100, help='topic size (default: 100)')
# Training options
parser.add_argument('--epoch', type=int, default=500, help='number of epochs to train WeTe (default: 500)')
parser.add_argument('--batchsize', type=int, default=500, help='batch size (default: 500)')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate for Adam to train WeTe (default: 1e-3)')
parser.add_argument('--test_every', type=int, default=10, help='do test every test_every epoches (default: 10)')
parser.add_argument('--beta', type=float, default=0.5, help='balance coefficient for bidirectional transport cost (default: 0.5)')
parser.add_argument('--epsilon', type=float, default=1.0, help='trade-off between transport cost and likelihood (default: 1.0)')
parser.add_argument('--device', type=str, default="0", help='which device for training: 0, 1, 2, 3 (GPU) or cpu')
parser.add_argument('--init_alpha', type=bool, default=True, help='Using K-means to initialise topic embeddings or not, setting to True will make faster and better performance.')
parser.add_argument('--train_word', type=bool, default=True, help='Finetuning word embedding or not, seting to True will make better performance.')
# Dataset options
parser.add_argument('--dataname', type=str, default='20ng_6', help='Dataset: 20ng_6|20ng_20|reuters|rcv2|web|tmn|dp')

# Other options
parser.add_argument('--glove', type=str, default="./glove.6B/glove.6B.100d.txt", help='load pretrained word embedding')
# parser.add_argument('--glove', type=str, default=None, help='load pretrained word embedding')
parser.add_argument('--every_test', type=int, default=5, help='test every test_num epoch')
parser.add_argument('--save_phi', type=int, default=10, help='save phi every save_phi epoch')
parser.add_argument('--save_path', type=str, default='WeTe_result', help='path for saving topics')

args = parser.parse_args()
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
args.device = device


if __name__ == '__main__':

    # Load dataset
    if args.dataname == '20ng_6':
        data_path = './dataset/20ng.pkl'
        args.clc_num = 20
    elif args.dataname == '20ng_20':
        data_path = './dataset/20ng.pkl'
        args.clc_num = 20
    elif args.dataname == 'reuters':
        data_path = './dataset/reuters.pkl'
        args.clc_num = 20
    elif args.dataname == 'rcv2':
        data_path ='dataset/rcv2.pkl'
        args.clc_num = 52
    elif args.dataname == 'web':
        data_path = './dataset/web.pkl'
        args.clc_num = 20
    elif args.dataname == 'tmn':
        data_path = './dataset/tmn.pkl'
        args.clc_num = 20
    elif args.dataname == 'dp':
        data_path = './dataset/dp.pkl'
        args.clc_num = 20
    else:
        print(f'Unknown dataset: {args.dataname}')
        exit()
    train_loader, voc = dataloader(data_path, dataname=args.dataname, mode='train', batch_size=args.batchsize)
    test_loader, _ = dataloader(data_path, dataname=args.dataname, mode='test', batch_size=args.batchsize)
    args.vocsize = len(voc)
    print(f'=============================   Setting   =============================\n {args}')
    print(f'len train: {len(train_loader)}, len test: {len(test_loader)}, voc_len: {len(voc)}')

    model = WeTe(args, voc=voc)
    trainer = Trainer(args, model, voc=voc)
    trainer.train(train_loader, test_loader)