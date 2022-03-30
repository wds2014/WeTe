#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
#                    _          _
#                .__(.)<  ??  >(.)__.
#                 \___)        (___/ 
# @Time    : 2022/3/28 下午10:14
# @Author  : wds -->> hellowds2014@gmail.com
# @File    : Trainer.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import time
import os
from tqdm import tqdm
from Utils import *
from cluster_clc import cluster_clc

class Trainer(object):
    """
    Trainer for WeTe
    """
    def __init__(self, args, model, voc=None):
        super(Trainer, self).__init__()
        self.model = model.to(args.device)
        self.epoch = args.epoch
        self.data_name = args.dataname
        self.device = args.device
        self.topic_k = args.K
        self.test_every = args.test_every
        self.train_num = -1
        self.clc_num = args.clc_num

        log_str = 'runs/{}/k_{}'.format(args.dataname, self.topic_k)
        now = int(round(time.time() * 1000))
        now_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(now / 1000))
        self.log_str = log_str + '/' + now_time
        if not os.path.exists(self.log_str):
            os.makedirs(self.log_str)

        self.trainable_params = []
        print('WeTe learnable params:')
        for name, params in self.model.named_parameters():
            if params.requires_grad:
                print(name)
                self.trainable_params.append(params)
        self.optimizer = torch.optim.Adam(self.trainable_params, lr=args.lr, weight_decay=1e-3)

    def train(self, train_loader, test_loader):
        for epoch in range(self.epoch):
            tr_loss = []
            tr_forward_cost = []
            tr_backward_cost = []
            tr_tm = []
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            self.model.train()
            for j, (bow, label) in pbar:
                self.train_num += 1
                train_data = to_list(bow.long(), device=self.device)
                bow = bow.to(self.device).float()
                loss, forward_cost, backward_cost, tm_loss, _ = self.model(train_data, bow)
                self.optimizer.zero_grad()
                loss.backward()
                for p in self.trainable_params:
                    try:
                        p.grad = p.grad.where(~torch.isnan(p.grad), torch.tensor(0., device=p.grad.device))
                        p.grad = p.grad.where(~torch.isinf(p.grad), torch.tensor(0., device=p.grad.device))
                        nn.utils.clip_grad_norm_(p, max_norm=20, norm_type=2)
                    except:
                        pass
                self.optimizer.step()
                tr_loss.append(loss.item())
                tr_forward_cost.append(forward_cost.item())
                tr_backward_cost.append(backward_cost.item())
                tr_tm.append(tm_loss.item())
                pbar.set_description(f'epoch: {epoch}|{self.epoch}, loss: {np.mean(tr_loss):.4f},  forword_cost: {np.mean(tr_forward_cost):.4f},  '
                                     f'backward_cost: {np.mean(tr_backward_cost):.4f}, TM_loss: {np.mean(tr_tm):.4f}')

            if epoch % self.test_every == 0:
                self.model.eval()
                train_theta = None
                train_label = None
                test_theta = None
                test_label = None
                tr_loss = []
                tr_forward_cost = []
                tr_backward_cost = []
                tr_tm = []
                te_loss = []
                te_forward_cost = []
                te_backward_cost = []
                te_tm = []
                with torch.no_grad():
                    ### visualize topics and save embeddings
                    phi = self.model.cal_phi()
                    vision_phi([phi.cpu().detach().numpy()], outpath=f'{self.log_str}/phi_{epoch}.txt', voc=self.model.voc)
                    self.model.save_embeddings(f'{self.log_str}/embeddings_{epoch}.pkl')

                    for j, (bow, label) in enumerate(train_loader):
                        train_data = to_list(bow.long(), device=self.device)
                        bow = bow.to(self.device).float()
                        loss, forward_cost, backward_cost, tm_loss, theta = self.model(train_data, bow)
                        tr_loss.append(loss.item())
                        tr_forward_cost.append(forward_cost.item())
                        tr_backward_cost.append(backward_cost.item())
                        tr_tm.append(tm_loss.item())
                        if train_theta is None:
                            train_theta = theta.cpu().detach().numpy()
                            train_label = label.detach().numpy()
                        else:
                            train_theta = np.concatenate((train_theta, theta.cpu().detach().numpy()))
                            train_label = np.concatenate((train_label, label.detach().numpy()))
                    for j, (bow, label) in enumerate(test_loader):
                        test_data = to_list(bow.long(), device=self.device)
                        bow = bow.to(self.device).float()
                        loss, forward_cost, backward_cost, tm_loss, theta = self.model(train_data, bow)
                        te_loss.append(loss.item())
                        te_forward_cost.append(forward_cost.item())
                        te_backward_cost.append(backward_cost.item())
                        te_tm.append(tm_loss.item())
                        if test_theta is None:
                            test_theta = theta.cpu().detach().numpy()
                            test_label = label.detach().numpy()
                        else:
                            test_theta = np.concatenate((test_theta, theta.cpu().detach().numpy()))
                            test_label = np.concatenate((test_label, label.detach().numpy()))
                purity_value, nmi_value, ami_value, micro_prec, micro_recall, micro_f1_score, acc = cluster_clc(train_theta, train_label, test_theta, test_label, self.clc_num)
                print(f'*************************** Test Summary **************************')
                print(f'Epoch {epoch}|{self.epoch}\n'
                      f'TRAIN, loss: {np.mean(tr_loss):.4f}, forward cost: {np.mean(tr_forward_cost):.4f}, backward cost: {np.mean(tr_backward_cost):.4f}, TM loss: {np.mean(tr_tm):.4f}\n'
                      f'TEST, loss: {np.mean(te_loss):.4f}, forward cost: {np.mean(te_forward_cost):.4f}, backward cost: {np.mean(te_backward_cost):.4f}, TM loss: {np.mean(te_tm):.4f}\n'
                      f'Clustering, purity: {purity_value:.4f}, nmi: {nmi_value:.4f}, ami: {ami_value:.4f}\n'
                      f'Classification, micro_p: {micro_prec:.4f}, micro_r: {micro_recall:.4f}, micro_f1: {micro_f1_score:.4f}')
