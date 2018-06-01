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
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

import time
# time.sleep(3600*1.5)

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


def lgb_prepare(dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                feval=None, early_stopping_rounds=50, num_boost_round=3000, verbose_eval=10,
                categorical_features=None):
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
    # dtrain = None
    # dlabels = None
    gc.collect()

    vlabels = dvalid[target].values
    dvalid = dvalid[predictors].values.astype(np.float32)
    gc.collect()
    xgvalid = lgb.Dataset(dvalid, label=vlabels,
                          feature_name=predictors,
                          categorical_feature=categorical_features,
                          free_raw_data=False
                          )
    # dvalid = None
    # dlabels = None
    gc.collect()

    return xgtrain, xgvalid


def lgb_train(params, xgtrain, xgvalid, predictors, target='target', objective='binary', metrics='auc',
              feval=None, early_stopping_rounds=50, num_boost_round=3000, verbose_eval=10):
    evals_results = {}

    bst1 = lgb.train(params,
                     xgtrain,
                     valid_sets=[xgvalid],
                     valid_names=['valid'],
                     evals_result=evals_results,
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10)

    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics + ":", evals_results['valid'][metrics][bst1.best_iteration - 1])

    return (bst1, bst1.best_iteration, evals_results['valid'][metrics][bst1.best_iteration - 1])


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

    def_params = {
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
        'scale_pos_weight': 200  # because training data is extremely unbalanced
    }


    def get_params(space):
        tmp_paras = def_params
        tmp_paras.update(space)
        tmp_paras['min_data_in_leaf'] = int(tmp_paras['min_data_in_leaf'])
        tmp_paras['num_leaves'] = int(tmp_paras['num_leaves'])
        tmp_paras['max_depth'] = int(tmp_paras['max_depth'])

        return tmp_paras


    print "params", def_params
    target = 'is_attributed'

    categorical = list(set(['app', 'device', 'os', 'channel', 'hour']) & set(predictors))

    log = []

    xgtrain, xgvalid = lgb_prepare(train_df,
                                   val_df,
                                   predictors,
                                   target,
                                   categorical_features=categorical)


    def score(params):
        global train_df, val_df, predictors, target, categorical
        print("Training with params: ")
        print(get_params(params))

        (bst, best_iteration, score) = lgb_train(get_params(params),
                                                 xgtrain,
                                                 xgvalid,
                                                 predictors,
                                                 target,
                                                 objective='binary',
                                                 metrics='auc',
                                                 early_stopping_rounds=30,
                                                 verbose_eval=True,
                                                 num_boost_round=1000)

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
        print("\tScore {0}\n\n".format(score))

        log.append([score, best_iteration, params])
        print sorted(log, key=lambda x: -x[0])[:10]
        # The score function should return the loss (1-score)
        # since the optimize function looks for the minimum
        loss = 1 - score
        return {'loss': loss, 'status': STATUS_OK}


    def optimize(
            # trials,
            random_state=2017):
        """
        This is the optimization function that given a space (space here) of
        hyperparameters and a scoring function (score here), finds the best hyperparameters.
        """
        # To learn more about XGBoost parameters, head to this page:
        # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
        space = {
            # 'boosting_type': hp.choice( 'boosting_type', ['gbdt', 'dart' ] ),
            'max_depth': hp.quniform("max_depth", 4, 9, 1),
            'num_leaves': hp.quniform('num_leaves', 10, 100, 1),
            'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 150, 1),
            'feature_fraction': hp.uniform('feature_fraction', 0.4, 1.0),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 1.0),
            'learning_rate': hp.loguniform('learning_rate', -2.5, -1.6),
            'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 4),
            # "scale_pos_weight": hp.loguniform('scale_pos_weight', 3, 10),
            # 'lambda_l1': hp.uniform('lambda_l1', 1e-4, 1e-6 ),
            # 'lambda_l2': hp.uniform('lambda_l2', 1e-4, 1e-6 ),
            'seed': hp.randint('seed', 2000000)
        }
        trials = Trials()
        best = hyperopt.fmin(fn=score,
                             space=space,
                             algo=tpe.suggest,
                             max_evals=500,
                             trials=trials,
                             verbose=1)
        return best


    best_hyperparams = optimize(
            # trials
    )
    print("The best hyperparameters are: ", "\n")
    print(best_hyperparams)
