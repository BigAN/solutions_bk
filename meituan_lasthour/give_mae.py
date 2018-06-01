# encoding:utf8
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import gc
import time
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import argparse
from utils import nice_method as nm
import pandas as pd
# import sklearn.metrics
# from sklearn.metrics import f1_score, roc_auc_score
# from sklearn.model_selection import train_test_split
import lightgbm as lgb
import lgbm_cv as lc
import numpy as np

from sklearn.metrics import mean_absolute_error


def init_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--real', type=str, dest='real',
                        help="data")
    parser.add_argument('--prd', type=str, dest='prd',
                        help="data")
    return parser.parse_args()


if __name__ == '__main__':
    args = init_arguments()

    real = pd.read_csv(args.real)
    prd = pd.read_csv(args.prd)

    print "!!!!!!!!!!!!!! rs is {} !!!!!!!!!!!!!!!!!".format(
            mean_absolute_error(real["delivery_duration"], prd["delivery_duration"]))

    for i in [0, 1, 2, 3, 4, 5]:
        print i
        print "!!!!!!!!!!!!!! bin is {} rs is {} !!!!!!!!!!!!!!!!!". \
            format(i,
                   mean_absolute_error(real[real.delivery_duration_bin == i]["delivery_duration"],
                                       prd[real.delivery_duration_bin == i]["delivery_duration"]))
    print "prd"
    print prd["delivery_duration"].describe()
    print "actual"
    print real["delivery_duration"].describe()
