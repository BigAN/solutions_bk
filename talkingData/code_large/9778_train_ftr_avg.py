import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import argparse
import os
import scipy

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


class tick_tock:
    def __init__(self, process_name, verbose=1):
        self.process_name = process_name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print(self.process_name + " begin ......")
            self.begin_time = time.time()

    def __exit__(self, type, value, traceback):
        if self.verbose:
            end_time = time.time()
            print("*" * 50 + " START!!!! " + "*" * 50)
            print(self.process_name + " end ......")
            print(' time lapsing {0} s'.format(end_time - self.begin_time))
            print("#" * 50 + " END.... " + "#" * 50)


def init_arguments():
    def str_to_bool(s):
        if s == 't':
            return True
        elif s == 'f':
            return False
        else:
            raise ValueError

    parser = argparse.ArgumentParser()

    parser.add_argument('--tst', type=str, dest='test_df', default=True,
                        help="data")
    parser.add_argument('--tra', type=str, dest='train_df', default=False,
                        help="data")
    parser.add_argument('--val', type=str, dest="val_df", default=50000000,
                        help="map key(rider or rider_poi)")
    parser.add_argument('--fea', type=str, dest="feas", default=50000000,
                        help="map key(rider or rider_poi)")
    parser.add_argument('--rm', type=str, dest="rm_feas", default="",
                        help="map key(rider or rider_poi)")
    parser.add_argument('--addfeas', type=str, dest="add_feas", default="",
                        help="map key(rider or rider_poi)")
    parser.add_argument('--detailfeas', type=str, dest="detail_feas", default="",
                        help="map key(rider or rider_poi)")
    parser.add_argument('--fileno', type=str, dest="fileno", default="basic",
                        help="map key(rider or rider_poi)")

    return parser.parse_args()


def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                      feval=None, early_stopping_rounds=50, num_boost_round=3000, verbose_eval=10,
                      categorical_features=None, sub=None, test_df=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'nthread': 12,
        # 'verbose': 10,
        'learning_rate': 0.10,
        'num_leaves': 20,  # > 2^max_depth - 1
        'max_depth': 7,  # -1 means no limit
        'min_data_in_leaf': 10,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 1023,  # Number of bucketed bin for feature values
        'subsample': 0.9,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.75,  # Subsample ratio of columns when constructing each tree.
        'min_sum_hessian_in_leaf': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight': 200,  # because training data is extremely unbalanced
        'nthread': 12,
        'verbose': 0,
    }

    lgb_params.update(params)
    print lgb_params

    print("preparing validation datasets")

    def get_values(df, type):
        n = len(df)
        d1 = df[:, :n / 2].astype('float')
        d2 = df[:, n / 2:].astype('float')
        data = np.hstack(d1, d2)
        return data

    dlabels = dtrain[target].values
    dtrain = dtrain[predictors].values.astype(np.float32)
    gc.collect()
    xgtrain = lgb.Dataset(dtrain, label=dlabels,
                          feature_name=predictors,
                          categorical_feature=categorical_features,
                          free_raw_data=False

                          )
    del dtrain
    del dlabels
    gc.collect()

    vlabels = dvalid[target].values
    dvalid = dvalid[predictors].values.astype(np.float32)
    # gc.collect()
    xgvalid = lgb.Dataset(dvalid, label=vlabels,
                          feature_name=predictors,
                          categorical_feature=categorical_features,
                          free_raw_data=False

                          )
    del dvalid
    del vlabels
    gc.collect()

    evals_results = {}
    xgtest = test_df[predictors].values.astype('float32')
    N = 10
    sub['is_attributed'] = 0
    for i in range(N):
        lgb_params['seed'] = lgb_params['seed'] + i
        bst1 = lgb.train(lgb_params,
                         xgtrain,
                         valid_sets=[xgvalid],
                         valid_names=['valid'],
                         evals_result=evals_results,
                         num_boost_round=num_boost_round,
                         early_stopping_rounds=early_stopping_rounds,
                         verbose_eval=10,
                         feval=feval)

        print("\nModel Report")
        print("bst1.best_iteration: ", bst1.best_iteration)
        print(metrics + ":", evals_results['valid'][metrics][bst1.best_iteration - 1])
        sub['is_attributed'] += bst1.predict(xgtest, num_iteration=bst1.best_iteration)
    sub['is_attributed'] = sub['is_attributed'] / N
    return sub


with tick_tock("start read"):
    args = init_arguments()
    fileno = args.fileno

    print args
    print "args.test_df", args.test_df
    test_df = pd.read_feather(args.test_df)
    val_df = pd.read_feather(args.val_df)
    train_df = pd.read_feather(args.train_df)

    print "all cols", train_df.columns
    rm_feas = args.rm_feas.split(",")
    predictors = args.feas.split(",")
    add_feas_files = args.add_feas.split(",")
    detail_feas_list = args.detail_feas.split(",")

    print("predictors : ", predictors, len(predictors))
    predictors = list(set(args.feas.split(",")) - set(rm_feas))
    print 'after rm ', predictors, len(predictors)


    def check_join(a, b):
        if len(a) != len(b):
            print len(a), len(b)
            raise Exception("not match {},{}".format(len(a), len(b)))


    with tick_tock("start add feature"):
        for n, af in enumerate(add_feas_files):
            print af, "af"
            if len(af) > 3:
                be_added = detail_feas_list[n].split("#")
                print 'add fea:', be_added
                to_append = pd.read_feather("train_" + af + "_119903890_179903890_False_False.ftr")
                print be_added, "be_added"
                be_added = be_added if len(be_added) > 0 and len(be_added[0]) > 0 else list(to_append.columns)
                be_added = list(set(be_added) - set([u'minute', u'second']))
                gc.collect()
                check_join(train_df, to_append)
                train_df = train_df.join(to_append[be_added])
                gc.collect()

                to_append = pd.read_feather("test_" + af + "_119903890_179903890_False_False.ftr")[be_added]
                check_join(test_df, to_append)
                test_df = test_df.join(to_append[be_added])
                gc.collect()
                to_append = pd.read_feather("val_" + af + "_119903890_179903890_False_False.ftr")[be_added]
                check_join(val_df, to_append)
                val_df = val_df.join(to_append)

                print "join with af", af, "after join ", train_df.columns
                del to_append
                gc.collect()

                predictors.extend(be_added)

    print "rm_feas", rm_feas
    predictors = list(set(predictors) - set(rm_feas))

    # print predictors
    print("\ntrain size: ", len(train_df))
    print("\nvalid size: ", len(val_df))
    print("\ntest size : ", len(test_df))

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    gc.collect()

with tick_tock("Training..."):
    print "final feature is :", predictors
    print "print head 5 ", train_df.head(5)
    print "print tail 5 ", train_df.tail(5)
    print "print val 5 ", val_df.head(5)
    print "print val 5 ", val_df.tail(5)

    # params = {
    #     'learning_rate': 0.10,
    #     # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
    #     'num_leaves': 20,  # > 2^max_depth - 1
    #     'max_depth': 7,  # -1 means no limit
    #     'min_child_samples': 10,  # Minimum number of data need in a child(min_data_in_leaf)
    #     'max_bin': 1023,  # Number of bucketed bin for feature values
    #     'subsample': 0.8,  # Subsample ratio of the training instance.
    #     'subsample_freq': 3,  # frequence of subsample, <=0 means no enable
    #     'colsample_bytree': 0.75,  # Subsample ratio of columns when constructing each tree.
    #     'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    #     'scale_pos_weight': 200,  # because training data is extremely unbalanced
    #     'seed': 2007
    # }
    params = {'num_leaves': 63, 'scale_pos_weight': 200, 'min_sum_hessian_in_leaf': 30.935343666953045,
              'learning_rate': 0.12151879193655869, 'seed': 1042138, 'min_data_in_leaf': 27,
              'bagging_fraction': 0.8266534855846595, 'max_depth': 7, 'feature_fraction': 0.5876703044209775}
    # params = {'num_leaves': 39, 'scale_pos_weight': 994.8501651976337, 'min_sum_hessian_in_leaf': 1.2591696749327386,
    #           'learning_rate': 0.01450361597313792, 'seed': 1708401, 'min_data_in_leaf': 68,
    #           'bagging_fraction': 0.8039432625290821, 'max_depth': 5, 'feature_fraction': 0.6199369631079862}
    print "params", params
    target = 'is_attributed'

    categorical = list(set(['app', 'device', 'os', 'channel', 'hour']) & set(predictors))

    sub = lgb_modelfit_nocv(params,
                            train_df,
                            val_df,
                            predictors,
                            target,
                            objective='binary',
                            metrics='auc',
                            early_stopping_rounds=30,
                            verbose_eval=True,
                            num_boost_round=1500,
                            categorical_features=categorical, sub=sub, test_df=test_df)

    with tick_tock("predicting..."):
        print("Predicting...")
        # val_df['prd'] = bst.predict(val_df[predictors].values.astype('float32'), num_iteration=best_iteration)
        #     if not debug:
        #         print("writing...")
        sub.to_csv('sub_it%s' % (fileno) + '.csv.gz', index=False, float_format='%.9f', compression='gzip')
        # sub.to_csv('sub_it%d.csv' % (fileno), index=False, float_format='%.9f')
        print("done...")  # return sub
