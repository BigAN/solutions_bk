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
import xgboost as xgb
import lgbm_cv as lc
from sklearn.metrics import mean_absolute_error


# features = list((set(basic_features) | set(advanced_features)) - set(to_drop) - set(fea_drop))


# features = single_feas

def init_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--label', type=str, dest='label',
                        help="data")

    parser.add_argument('--train', type=str, dest='train_path',
                        help="data")
    parser.add_argument('--test', type=str, dest='test_path',
                        help="data")
    parser.add_argument('--round', type=int, dest='round',
                        help="data", default=500)

    parser.add_argument('--output', type=str, dest='out_path',
                        help="data")
    return parser.parse_args()


if __name__ == '__main__':
    args = init_arguments()

    train_data = lc.deco_data(pd.read_csv(args.train_path, sep="\t", skiprows=1))
    test_data = lc.deco_data(pd.read_csv(args.test_path, sep="\t", skiprows=1))
    features = list(set(train_data.columns.tolist()) - set(lc.to_drop) - set(lc.fea_drop))
    print "features,length {}, {}".format(len(features), features)

    labels = train_data[args.label].values.astype(np.float32).flatten()
    labels_val = test_data[[args.label]].values.astype(np.float32).flatten()

    dtrain = xgb.DMatrix(train_data[features].values, labels, missing=np.nan)
    dtest = xgb.DMatrix(test_data[features].values, labels_val, missing=np.nan)

    print('training model...')
    watchlist = [(dtest, 'dtest')]
    # param = {
    #     'booster': 'gbtree',
    #     'objective': 'binary:logistic',
    #     'eval_metric': 'auc',
    #     'eta': 0.1,
    #     'num_round': args.round,
    #     'colsample_bytree': 0.3,
    #     "colsample_bylevel": 0.3,
    #     'subsample': 0.7,
    #     "min_child_weight": 50,
    #     "max_delta_step": 0.3,
    #     'max_depth': 5,
    #     'nthread': -1,
    #     'seed': 20171001,
    #     'silent': 1,
    # }
    # param = {'num_leaves': 100, 'task': 'train', 'verbose': 1, 'learning_rate': 0.01, 'nthread': 8,
    #               'min_data_in_leaf': 35, 'objective': 'binary', 'boosting_type': 'gbdt', 'metric': [ 'auc'],
    #               'feature_fraction': 0.1, "bagging_fraction": 0.7, "bagging_freq": 5}
    # param = {
    #     'colsample_bytree': 0.3, 'silent': 1, 'eval_metric': 'auc', 'colsample_bylevel': 0.6000000000000001,
    #     'max_delta_step': 0.2, 'nthread': 8, 'min_child_weight': 35.0, 'n_estimators': 200,
    #     'subsample': 0.7000000000000001, 'eta': 0.1, 'objective': 'binary:logistic', 'seed': 2017, 'max_depth': 10,
    #     'gamma': 0.8500000000000001, 'booster': 'gbtree',
    #     'eval_metric': 'auc', 'nthread': 8, 'silent': 1
    # }
    param = {'booster': 'gbtree',
             'colsample_bylevel': 0.7,
             'colsample_bytree': 0.5,
             'eta': 0.15,
             'eval_metric': 'auc',
             'gamma': 0.5,
             'max_delta_step': 0.2,
             'max_depth': 7,
             'min_child_weight': 10.0,
             'nthread': 8,
             'objective': 'binary:logistic',
             'seed': 2017,
             'silent': 1,
             'subsample': 0.75}

    bst = xgb.train(param, dtrain, args.round, watchlist, verbose_eval=10, early_stopping_rounds=30)

print ">>>>>", bst.best_score
print('generating prediction...')
pred = bst.predict(dtest)

test_data.loc[:, "prd"] = pred

# print mean_absolute_error(test_data[args.label], test_data["delivery_duration_prd"])
# print test_data["delivery_duration_prd"].describe()
# print test_data[args.label].describe()
# test_data['prd'] = gbm.predict(test_data[features])
import os

# ax = lgb.plot_tree(gbm,tree_index=8, figsize=(20, 8), show_info=['split_gain'])
# plt.show()
# test_data.to_csv(args.out_path)


# rs = order_id.merge(test_data[["order_id", "delivery_duration"]], left_on="order_id", right_on="order_id",
#                     how="left")
#
# rs.to_csv(args.out_path, header=['order_id', 'delivery_duration'], index=False)
