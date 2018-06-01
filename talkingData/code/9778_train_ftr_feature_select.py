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
                      categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric': metrics,
        'learning_rate': 0.05,
        # 'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        # "feature_fraction_seed": 42,
        # 'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
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

    xgtrain = lgb.Dataset(dtrain[predictors].values.astype(np.float32), label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values.astype(np.float32), label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

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

    return (bst1, bst1.best_iteration)


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
                to_append = pd.read_feather("train_" + af + "_149903890_184903890_False_False.ftr")
                print be_added, "be_added"
                be_added = be_added if len(be_added) > 0 and len(be_added[0]) > 0 else list(to_append.columns)
                be_added = list(set(be_added) - set([u'minute', u'second']))
                gc.collect()
                check_join(train_df, to_append)
                train_df = train_df.join(to_append[be_added])
                gc.collect()

                to_append = pd.read_feather("test_" + af + "_149903890_184903890_False_False.ftr")[be_added]
                check_join(test_df, to_append)
                test_df = test_df.join(to_append[be_added])
                gc.collect()
                to_append = pd.read_feather("val_" + af + "_149903890_184903890_False_False.ftr")[be_added]
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

rm_fea_l = []
rs = []
for rm_fea in [
                  # u'app',
                  u'channel', u'ip_os_device_app_nextClick', u'os', u'ip_device_by_channel_countuniq',
                  u'ip_appcount', u'ip_os_device_nextClick', u'ip_day_by_period_countuniq', u'ip_by_channel_countuniq',
                  u'ip_day_hourcount', u'ip_by_os_countuniq', u'ip_app_by_os_countuniq', u'ip_by_app_countuniq',
                  u'ip_device_os_by_app_countuniq', u'device', u'ip_app_device_os_channel_nextClick',
                  u'next_next_ip_os_device_app_diff', u'app_by_channel_countuniq', u'ip_device_os_by_app_cumcount',
                  u'next_prev_ip_os_device_app_diff', u'hour', u'ip_device_os_app_by_period_countuniq',
                  u'ip_by_device_countuniq', u'ip_app_by_device_countuniq', u'ip_day_by_hour_countuniq',
                  u'ip_os_prevClick', u'ip_channel_prevClick', u'ip_by_os_cumcount', u'ip_app_oscount',
                  u'ip_app_os_by_hour_var', "nothing"][::-1]:
    rm_fea_l.append(rm_fea)
    print "rm_fea_l is ", rm_fea_l
    predictors = list(set(predictors) - set(rm_fea_l))
    print "predictors after rm,", predictors
    with tick_tock("Training..."):
        params = {
            'learning_rate': 0.10,
            # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
            'num_leaves': 20,  # > 2^max_depth - 1
            'max_depth': 7,  # -1 means no limit
            'min_child_samples': 10,  # Minimum number of data need in a child(min_data_in_leaf)
            'max_bin': 1023,  # Number of bucketed bin for feature values
            'subsample': 0.8,  # Subsample ratio of the training instance.
            'subsample_freq': 3,  # frequence of subsample, <=0 means no enable
            'colsample_bytree': 0.75,  # Subsample ratio of columns when constructing each tree.
            'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
            'scale_pos_weight': 200  # because training data is extremely unbalanced
        }
        print "params", params
        target = 'is_attributed'

        categorical = list(set(['app', 'device', 'os', 'channel', 'hour', 'day', 'minute', 'second']) & set(predictors))

        (bst, best_iteration) = lgb_modelfit_nocv(params,
                                                  train_df,
                                                  val_df,
                                                  predictors,
                                                  target,
                                                  objective='binary',
                                                  metrics='auc',
                                                  early_stopping_rounds=20,
                                                  verbose_eval=True,
                                                  num_boost_round=10 ** 3,
                                                  categorical_features=categorical)

    # del train_df
    # del val_df
    gc.collect()

    with tick_tock("feature importance"):
        df = pd.DataFrame({'feature': bst.feature_name(), 'importances': bst.feature_importance('gain')})
        df['fscore'] = df['importances'] / df['importances'].sum()

        print(df.sort_values('importances', ascending=False).to_string(index=False))
        print("-" * 20 + " features" + "-" * 20)
        print(
            [x.strip() for x in
             df.sort_values('importances', ascending=False).feature.to_string(index=False).split("\n")])

    cur_s = bst.best_score['valid']['auc']
    rs.append([str(cur_s), ",".join(rm_fea_l)])
    print "rs,", rs

    # with tick_tock("predicting..."):
    #     print("Predicting...")
    #     sub['is_attributed'] = bst.predict(test_df[predictors], num_iteration=best_iteration)
    #     #     if not debug:
    #     #         print("writing...")
    #     sub.to_csv('sub_it%d' % (fileno) + "_{0}_".format(rm_fea) + '.csv.gz', index=False, float_format='%.9f',
    #                compression='gzip')
    #     # sub.to_csv('sub_it%d.csv' % (fileno), index=False, float_format='%.9f')
    #     print("done...")
    print "\n".join([":".join(x) for x in rs])
    print "#" * 150

    print "\n".join([":".join(x) for x in sorted(rs)])
    # return sub
