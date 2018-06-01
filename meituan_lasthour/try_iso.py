import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
import solutions.meituan_lasthour.stack.regression.feat_engineer as feg
import solutions.meituan_lasthour.lgbm_cv as lc
import math

from sklearn.metrics import mean_absolute_error

SEED = 2019

rf = RandomForestRegressor(n_estimators=100, max_depth=8, n_jobs=-1, random_state=SEED)
args = lc.init_arguments()
x_train, x_test, y_train, ntrain, ntest = feg.load_base_data(args.train_path, args.test_path)

y_train = y_train.ravel()

print(x_train.shape)


def get_oof_regressor_clb(clf, x_train, y_train, x_test, ntrain, ntest, NFOLDS=5):
    calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
    kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED, shuffle=True)
    oof_train = np.zeros((ntrain, 1))
    oof_test = np.zeros((ntest, 1))
    oof_test_skf = np.empty((NFOLDS, ntest, 1))
    rs = []
    print("Anthor base regressor model")
    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train.iloc[train_index]
        y_tr = y_train[train_index]
        print("clf-----" + str(x_tr.shape[1]))
        x_te = x_train.iloc[test_index]
        calibrated_clf.fit(x_tr, y_tr)
        oof_train[test_index] = clf.predict(x_te).reshape(-1, 1)
        print("--------->>>>>>>>:----r2 value :" + str(
            r2_score(y_train[test_index], oof_train[test_index])))  ## only to see the base model how well it perform
        print("--------->>>>>>>>:----mae :" + str(
            mean_absolute_error(y_train[test_index], oof_train[test_index])))  ## only to see the base model how well it perform
        rs.append(mean_absolute_error(y_train[test_index], oof_train[test_index]))
        oof_test_skf[i, :] = clf.predict(x_test).reshape(-1, 1)
    print "with cal,",rs
    print sum(rs)/len(rs)
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def get_oof_regressor(clf, x_train, y_train, x_test, ntrain, ntest, NFOLDS=5):
    # calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
    kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED, shuffle=True)
    oof_train = np.zeros((ntrain, 1))
    oof_test = np.zeros((ntest, 1))
    oof_test_skf = np.empty((NFOLDS, ntest, 1))
    rs = []
    print("Anthor base regressor model")
    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train.iloc[train_index]
        y_tr = y_train[train_index]
        print("clf-----" + str(x_tr.shape[1]))
        x_te = x_train.iloc[test_index]
        clf.fit(x_tr, y_tr)
        oof_train[test_index] = clf.predict(x_te).reshape(-1, 1)
        print("--------->>>>>>>>:----r2 value :" + str(
            r2_score(y_train[test_index], oof_train[test_index])))  ## only to see the base model how well it perform
        print("--------->>>>>>>>:----mae :" + str(
            mean_absolute_error(y_train[test_index], oof_train[test_index])))  ## only to see the base model how well it perform
        rs.append(mean_absolute_error(y_train[test_index], oof_train[test_index]))
        oof_test_skf[i, :] = clf.predict(x_test).reshape(-1, 1)

    print "without cal,",rs
    print sum(rs)/len(rs)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

get_oof_regressor(rf, x_train, y_train, x_test, ntrain, ntest)
get_oof_regressor_clb(rf, x_train, y_train, x_test, ntrain, ntest)

