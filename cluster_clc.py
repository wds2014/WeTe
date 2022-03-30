#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
#                    _          _
#                .__(.)<  ??  >(.)__.
#                 \___)        (___/ 
# @Time    : 2022/3/29 下午8:33
# @Author  : wds -->> hellowds2014@gmail.com
# @File    : cluster_clc.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.cluster import k_means
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score
import numpy as np

"""
perform K-means and logistic-regularization classification on train and test dataset
and return Purity, NMI for clustering
return P, R, F1, and ACC for classification
"""


def purity(labels, clustered):
    # find the set of cluster ids
    ### see http://www.cse.chalmers.se/~richajo/dit862/L13/Text%20clustering.html
    cluster_ids = set(clustered)
    N = len(clustered)
    majority_sum = 0
    for cl in cluster_ids:
        # for this cluster, we compute the frequencies of the different human labels we encounter
        # the result will be something like { 'camera':1, 'books':5, 'software':3 } etc.
        labels_cl = Counter(l for l, c in zip(labels, clustered) if c == cl)

        # we select the *highest* score and add it to the total sum
        majority_sum += max(labels_cl.values())

    # the purity score is the sum of majority counts divided by the total number of items
    return majority_sum / N


def normalization(data):
    _range = np.max(data, axis=1, keepdims=True) - np.min(data, axis=1, keepdims=True)
    return (data - np.min(data, axis=1, keepdims=True)) / _range


def standardization(data):
    mu = np.mean(data, axis=1, keepdims=True)
    sigma = np.std(data, axis=1, keepdims=True)
    return (data - mu) / sigma

def cluster_clc(train_data, train_label, test_data, test_label, clc_num):
    train_data_norm = standardization(train_data)
    test_data_norm = standardization(test_data)
    #### clustering
    tmp = k_means(test_data_norm, clc_num)
    predict_label = tmp[1]
    purity_value = purity(test_label, predict_label)
    nmi_value = normalized_mutual_info_score(test_label, predict_label)
    ami_value = adjusted_mutual_info_score(test_label, predict_label)

    ##### LR classifier
    clf = LogisticRegression(random_state=0, C=1.0, solver='liblinear', multi_class='ovr', n_jobs=-1).fit(
        train_data_norm, train_label)
    pred_label_list = list(clf.predict(test_data_norm))
    true_label_list = list(test_label)

    micro_prec, micro_recall, micro_f1_score, _ = precision_recall_fscore_support(true_label_list, pred_label_list,
                                                                                  average="micro")
    acc = accuracy_score(true_label_list, pred_label_list)
    return purity_value, nmi_value, ami_value, micro_prec, micro_recall, micro_f1_score, acc
