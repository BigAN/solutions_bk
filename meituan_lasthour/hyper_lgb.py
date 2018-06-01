# coding=UTF-8

import pandas as pd
import numpy as np
import utils as util
from hyperopt import fmin, hp, tpe
import hyperopt
from time import clock
from utils import *
from sklearn.metrics import mean_absolute_error
import lgbm_cv as lc
import lightgbm as lgb
import pickle
model_name = 'lgb'

def xgb_train(dtrain, dtest, param, offline=True, verbose=True, num_boost_round=1000):
    if verbose:
        model = lgb.train(param, dtrain, 
                        valid_sets=[dtest], valid_names=["valid_data"], 
                        num_boost_round=num_boost_round, verbose_eval=10, early_stopping_rounds=5)
        df = pd.DataFrame({'feature': gbm.feature_name(), 'importances': gbm.feature_importance()})
        df = df.sort_values('importances')
        if offline:
            feature_score_file = './hyper_lgb_log/offline_feature_score' + '.csv'
        else:
            feature_score_file = './hyper_lgb_log/online_feature_score' + '.csv'
        df.to_csv(feature_score_file)
    else:
        model = lgb.train(param, dtrain, num_boost_round=num_boost_round)
    return model

def xgb_predict(model, dtest):
    print 'model_best_ntree_limit : {0}\n'.format(model.best_iteration)
    pred_y = model.predict(dtest, num_iteration=model.best_iteration)
    return pred_y

def tune_xgb(train_data, test_data):
    dtrain = lgb.Dataset(train_data[features], labels)
    dtest = lgb.Dataset(test_data[features], labels_val, reference=dtrain)

    tune_reuslt_file = "./hyper_lgb_log/tune_" + model_name + ".csv"
    f_w = open(tune_reuslt_file, 'w')
    def objective(args):
        params = {
            'task': 'train',
            'boosting_type': args['boosting_type'],
            'objective': args['objective'],
            'metric': {"mae"},
            'num_leaves': int(args['num_leaves']),
            'min_sum_hessian_in_leaf': args['min_sum_hessian_in_leaf'],
            'min_data_in_leaf': int(args['min_data_in_leaf']),
            'max_depth': -1,
            'learning_rate': args['learning_rate'],
            'feature_fraction': args['feature_fraction'],
            'verbose': 1,
        }

        print "training..."
        print params
        model = xgb_train(dtrain, dtest, params, offline=True, verbose=False, num_boost_round=int(args['n_estimators']))
        model.save_model('dump_lgb_model_txt')
        print "predicting..."
        pred_y = xgb_predict(model, test_data[features])
        test_y = dtest.get_label()
        mae = mean_absolute_error(pred_y, test_y)

        xgb_log.write(str(args))
        xgb_log.write('\n')
        xgb_log.write(str(mae))
        xgb_log.write('\n')
        return mae

    # Searching space
    space = {
        #'boosting_type': hp.choice("boosting_type", ["gbdt","rf","dart","goss"]),
        #'objective': hp.choice("objective", ['regression_l2','regression_l1','huber','poisson','fair','regression']),
        'boosting_type': hp.choice("boosting_type", ["gbdt"]),
        'objective': hp.choice("objective", ['regression']),
        'num_leaves': hp.quniform("num_leaves", 100, 500, 5),
        'min_sum_hessian_in_leaf': hp.uniform("min_sum_hessian_in_leaf", 0.001, 10),
        'min_data_in_leaf': hp.quniform("min_data_in_leaf", 10, 100, 5),
        'learning_rate': hp.uniform("learning_rate", 0.01, 0.2),
        'feature_fraction': hp.uniform("feature_fraction", 0.5, 0.9),
        'n_estimators': hp.quniform("n_estimators", 50, 200, 5),
    }
    best_sln = fmin(objective, space, algo=tpe.suggest, max_evals=150)
    #best_sln = fmin(objective, space, algo=hyperopt.anneal.suggest, max_evals=300)
    pickle.dump(best_sln,f_w,True)
    mae = objective(best_sln)
    xgb_log.write(str(mae) + '\n')
    f_w.close()

def test(train_data, test_data,best_n_estimators):
    dtrain = lgb.Dataset(train_data[features], labels)
    dtest = lgb.Dataset(test_data[features], labels_val, reference=dtrain)

    final_result = "./hyper_lgb_log/lgb_online_result.csv"
    f_w = open(final_result, 'w')
    model = xgb_train(dtrain, dtest, init_params, offline, verbose=True,num_boost_round=best_n_estimators)
    pred_y = xgb_predict(model, test_data[features])
    test_y = dtest.get_label()
    mae = mean_absolute_error(pred_y, test_y)
    f_w.write(str(mae))
    f_w.close()

if __name__ == '__main__':
    t_start = clock()
    offline = True
    train_data = pd.read_csv('./data/train_cv.csv')
    test_data = pd.read_csv('./data/test_cv.csv')
    train_data.head(5)
    features = list(set(train_data.columns.tolist()[1:]) - set(lc.to_drop) - set(lc.fea_drop))

    labels = train_data['delivery_duration'].values.astype(np.float32).flatten()
    labels_val = test_data[['delivery_duration']].values.astype(np.float32).flatten()

    print 'Feature Dims : '
    print train_data.shape
    print test_data.shape

    if offline:
        xgb_log = open(name='./hyper_lgb_log/lgb_log.txt',mode='w')
        tune_xgb(train_data, test_data)
        xgb_log.close()
    else:
        tune_reuslt_file = "./hyper_lgb_log/tune_" + model_name + ".csv"
        f_w = open(tune_reuslt_file, 'r')
        args = pickle.load(f_w)
        f_w.close()

        init_params = {
            'task': 'train',
            'boosting_type': args['boosting_type'],
            'objective': args['objective'],
            'metric': {"mae"},
            'num_leaves': int(args['num_leaves']),
            'min_sum_hessian_in_leaf': args['min_sum_hessian_in_leaf'],
            'min_data_in_leaf': int(args['min_data_in_leaf']),
            'max_depth': -1,
            'learning_rate': args['learning_rate'],
            'feature_fraction': args['feature_fraction'],
            'verbose': 1,
        }
        test(train_data,test_data,args['n_estimators'])

    t_finish = clock()
    print('==============Costs time : %s s==============' % str(t_finish - t_start))
