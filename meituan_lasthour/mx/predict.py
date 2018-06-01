#coding=utf8
# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme 
# pylint: disable=superfluous-parens, no-member, invalid-name
import os
import re
import gc
import time
import pprint
import pickle
import hashlib
from math import log
import sys, datetime, math, random
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from io import BytesIO
from collections import namedtuple
from sklearn import metrics
import logging
from train_subsample import *


def predict(test_file, prefix, epoch):
    mod = mx.mod.Module.load(prefix, epoch, load_optimizer_states=True, data_names=['cx','dx'], label_names=[output_name], context=mx.gpu())
    #net = get_net(num_vocab, num_embed)
    #加载测试数据
    np_data = load_np_data_bin(test_file, True)
    mean, var, np_data = preprocess(np_data)
    data_iter = mx.io.NDArrayIter(data={'cx':np_data[:,1:CATE_FEA_LEN+1].astype(int), 'dx':np_data[:,CATE_FEA_LEN+1:]}, label={'y':np_data[:,0]}, batch_size=batch_size, shuffle=False)
    mod.bind(data_shapes=data_iter.provide_data, label_shapes=data_iter.provide_label)
    ys_eval=[]
    ps_eval=[]
    for pred, i_batch, batch in mod.iter_predict(data_iter):
        #print pred, i_batch, batch.data, batch.label
        np_pred = pred[0].asnumpy()
        np_label = batch.label[0].asnumpy()
        for i in range(np_pred.shape[0]):
            if np_pred.shape[1] == 2:#softmax
                p = np_pred[i][1]
            else:#sigmoid
                p = np_pred[i]
            if SAMPLE_RATE < 1.0:
                p = modify_pctr(p)
            y = np_label[i]
            ps_eval.append(p)
            ys_eval.append(y)
    fpr, tpr, thresholds = metrics.roc_curve(ys_eval, ps_eval, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    loss= logloss(ys_eval, ps_eval, 1.0)
    m = mae(ys_eval, ps_eval)
    print auc, loss, m


if __name__ == '__main__':
    predict("/data/xql/dense/dnn_dense_data_20170603", "checkpoint", 1)
