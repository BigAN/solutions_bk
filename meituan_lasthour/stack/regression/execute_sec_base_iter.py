import feat_engineer_sec as feg
import stack_model as sm
import config_params as cp

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

## Regression Model We will Use
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso

import solutions.meituan_lasthour.lgbm_cv as lc
from sklearn.model_selection import RandomizedSearchCV

SEED = 2018
args = lc.init_arguments()
x_train, x_test, y_train, ntrain, ntest = feg.load_base_data(args.train_path, args.test_path)

y_train = y_train.ravel()

print(x_train.shape)

# ---------------------------------------get_oof_tree(clf, x_train, y_train, x_test, ntrain, ntest, NFOLDS = 5)------------------------------------------------------

rf = RandomForestRegressor(n_estimators=500, max_depth=8, n_jobs=-1, random_state=SEED)
ada = AdaBoostRegressor(n_estimators=80, learning_rate=0.004, loss='square', random_state=SEED)
gb = GradientBoostingRegressor(learning_rate=0.01, n_estimators=120, subsample=0.921, max_depth=5, random_state=SEED)
gb10 = GradientBoostingRegressor(learning_rate=0.02, n_estimators=120, loss="huber", subsample=0.621, max_depth=10,
                                 random_state=SEED)
et = ExtraTreesRegressor(n_estimators=150, max_depth=6, max_features='sqrt', n_jobs=- 1, random_state=SEED)
le = LinearRegression(n_jobs=-1)
el = ElasticNet(random_state=SEED)
rd = Ridge(random_state=SEED)
ls = Lasso(random_state=SEED)
print "#" * 50


def gene_name(x):
    return "_".join(map(lambda x: str(x), x))


para_set = (
    ("sec_lg1",
     sm.get_oof_lgb,
     (cp.lgb_paras_regressor(5, "regression", round=200, sum_hessian=100), x_train, y_train, x_test, ntrain, ntest, 5)),
    ("sec_lg2",
     sm.get_oof_lgb, (
     cp.lgb_paras_regressor(3, "regression", round=300, sum_hessian=50, ), x_train, y_train, x_test, ntrain, ntest, 5)),
    ("sec_lg3",
     sm.get_oof_lgb,
     (cp.lgb_paras_regressor(7, "poisson", round=150, sum_hessian=5), x_train, y_train, x_test, ntrain, ntest, 5)),

)

print "#" * 50


def go(x):
    print "start :", x[0]

    rs = x[1](*x[2])
    x_train = pd.DataFrame(np.hstack([rs[0]]))
    x_test = pd.DataFrame(np.hstack([rs[1]]))
    x_train.columns = [x[0]]
    x_test.columns = [x[0]]
    print "#" * 59
    print "name:{}, tocsv".format(x[0])
    x_train.to_csv("stack_data/train_iter_{}_{}.csv".format(x[0], str(800)), index=None)
    x_test.to_csv("stack_data/test_iter_{}_{}.csv".format(x[0], str(800)), index=None)
    print "done"


import multiprocessing as mp

pool = mp.Pool(8)
pool.map(go, para_set)
