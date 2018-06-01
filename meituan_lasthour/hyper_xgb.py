# coding=UTF-8

import pandas as pd
import xgboost as xgb
import numpy as np
import utils as util
from hyperopt import fmin, hp, tpe
import hyperopt
from time import clock
from utils import *
from sklearn.metrics import mean_absolute_error
import lgbm_cv as lc

model_name = 'xgb'

def xgb_train(dtrain, dtest, param, offline=True, verbose=True, num_boost_round=1000):
    if verbose:
        if offline:
            watchlist = [(dtrain, 'train'), (dtest, 'test')]
        else:
            watchlist = [(dtrain, 'train')]
        model = xgb.train(param, dtrain, num_boost_round=num_boost_round, evals=watchlist, verbose_eval=10)
        feature_score = model.get_fscore()
        feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
        fs = []
        for key, value in feature_score:
            fs.append("{0},{1}\n".format(key, value))
        if offline:
            feature_score_file = './hyper_xgb_log/offline_feature_score' + '.csv'
        else:
            feature_score_file = './hyper_xgb_log/online_feature_score' + '.csv'
        f = open(feature_score_file, 'w')
        f.writelines("feature,score\n")
        f.writelines(fs)
        f.close()
    else:
        model = xgb.train(param, dtrain, num_boost_round=num_boost_round)
    return model

def xgb_predict(model, dtest):
    print 'model_best_ntree_limit : {0}\n'.format(model.best_ntree_limit)
    pred_y = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    return pred_y

def tune_xgb(dtrain, dtest):
    tune_reuslt_file = "./hyper_xgb_log/tune_" + model_name + ".csv"
    f_w = open(tune_reuslt_file, 'w')
    def objective(args):
        params = {
            'booster': 'gbtree',
            'objective': 'reg:linear',
            'eval_metric': args['eval_metric'],
            'eta': args['eta'],
            'gamma': args['gamma'],
            'colsample_bytree': args['colsample_bytree'],
            'max_depth': int(args['max_depth']),
            'subsample': args['subsample'],
            'silent' : 1
        }
    
        print "training..."
        print params
        model = xgb_train(dtrain, dtest, params, offline=True, verbose=True, num_boost_round=int(args['n_estimators']))
        model.dump_model('dump_model_txt')
        print "predicting..."
        pred_y = xgb_predict(model, dtest)
        test_y = dtest.get_label()
        mae = mean_absolute_error(pred_y, test_y)

        xgb_log.write(str(args))
        xgb_log.write('\n')
        xgb_log.write(str(mae))
        xgb_log.write('\n')
        return mae

    # Searching space
    space = {
        'eval_metric' : hp.choice('eval_metric', ['mae']),
        'n_estimators': hp.quniform("n_estimators", 100, 200, 10),
        'gamma': hp.loguniform("gamma", np.log(0.1), np.log(100)),
        'eta': hp.uniform("eta", 0.05, 0.15),
        'max_depth': hp.quniform("max_depth", 5, 10, 1),
        'subsample': hp.uniform("subsample", 0.5, 0.9),
        'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 0.9),
    }
    best_sln = fmin(objective, space, algo=tpe.suggest, max_evals=150)
    #best_sln = fmin(objective, space, algo=hyperopt.anneal.suggest, max_evals=300)
    pickle.dump(best_sln,f_w,True)
    mae = objective(best_sln)
    xgb_log.write(str(mae) + '\n')
    f_w.close()

def test(dtrain, dtest,best_n_estimators):
    final_result = "./hyper_xgb_log/xgb_online_result.csv"
    f_w = open(final_result, 'w')
    model = xgb_train(dtrain, dtest, init_params, offline, verbose=True,num_boost_round=best_n_estimators)
    pred_y = xgb_predict(model, dtest)
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

    dtrain = xgb.DMatrix(train_data[features].values, labels, missing=np.nan)
    dtest = xgb.DMatrix(test_data[features].values, labels_val, missing=np.nan)

    print 'Feature Dims : '
    print train_data.shape
    print test_data.shape

    if offline:
        xgb_log = open(name='./hyper_xgb_log/xgb_log.txt',mode='w')
        tune_xgb(dtrain, dtest)
        xgb_log.close()
    else:
        tune_reuslt_file = "./hyper_xgb_log/tune_" + model_name + ".csv"
        f_w = open(tune_reuslt_file, 'r')
        tune_xgb = pickle.load(f_w)
        f_w.close()

        best_n_estimators = int(tune_xgb['n_estimators'])
        best_eta = tune_xgb['eta']
        best_max_depth = int(tune_xgb['max_depth'])
        best_subsample = tune_xgb['subsample']
        best_colsample_bytree = tune_xgb['colsample_bytree']
        best_eval_metric = tune_xgb['eval_metric']
        best_gamma = tune_xgb['gamma']

        init_params = {
            'booster': 'gbtree',
            'objective': 'reg_linear',
            'eval_metric': best_eval_metric,
            'max_depth': best_max_depth,
            'subsample': best_subsample,
            'colsample_bytree': best_colsample_bytree,
            'eta': best_eta,
            'gamma': best_gamma
        }
        test(dtrain,dtest,best_n_estimators)

    t_finish = clock()
    print('==============Costs time : %s s==============' % str(t_finish - t_start))
