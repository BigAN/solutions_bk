import feat_engineer as feg
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
import random
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


'''
base,
poi_agg
cluster,
cur_block,
area_id_area_id#hour#minute,
'''

base_train = lc.gene_xtrain(x_train, ["base"])
best_train = lc.gene_xtrain(x_train, ["base", "poi_agg", "area_id_area_id#hour#minute"])
# train_1 = lc.gene_xtrain(x_train, ["base", "poi_agg", "area_id_area_id#hour#minute", "cur_block"])
# train_2 = lc.gene_xtrain(x_train, ["base", "poi_agg", "area_id_area_id#hour#minute", "cluster"])
# train_3 = lc.gene_xtrain(x_train, ["base", "area_id_area_id#hour#minute", "cluster"])
# train_4 = lc.gene_xtrain(x_train, ["poi_agg", "cluster"])
# train_5 = lc.random_gene_train(x_train, 80)
def rule1(x):
    return x.delivery_duration_bin >=4
def rule2(x):
    return (x.delivery_duration_bin >=2)&(x.delivery_duration_bin <4)
def rule3(x):
    return (x.delivery_duration_bin <2)
def rall(x):
    return  x.delivery_duration_bin >=-1

para_set = (
    # ("xg3", sm.get_oof_xgb_linear, (cp.xgb_params_level_5, x_train, y_train, x_test, ntrain, ntest, 4)),
    # ("xg8", sm.get_oof_xgb_linear, (cp.xgb_params_level_8, x_train, y_train, x_test, ntrain, ntest, 4)),
    # ("xg10", sm.get_oof_xgb_linear, (cp.xgb_params_level_10, x_train, y_train, x_test, ntrain, ntest, 4)),
    ("r1",
     sm.get_oof_lgb,
     (cp.lgb_paras_regressor(7, "regression", 1500), x_train, y_train, x_test, ntrain, ntest, 5, best_train, rule1)),
    ("r2",
     sm.get_oof_lgb,
     (cp.lgb_paras_regressor(7, "regression", 1500), x_train, y_train, x_test, ntrain, ntest, 5, best_train, rule2)),
    ("r3",
     sm.get_oof_lgb,
     (cp.lgb_paras_regressor(7, "regression", 1500), x_train, y_train, x_test, ntrain, ntest, 5, best_train, rule3)),
    ("rall",
     sm.get_oof_lgb,
     (cp.lgb_paras_regressor(7, "regression", 1500), x_train, y_train, x_test, ntrain, ntest, 5, best_train, rall)),
    ("rall2",
     sm.get_oof_lgb,
     (cp.lgb_paras_regressor(10, "regression", 1500), x_train, y_train, x_test, ntrain, ntest, 5, best_train, rall)),
    ("rall3",
     sm.get_oof_lgb,
     (cp.lgb_paras_regressor(5, "regression", 1500), x_train, y_train, x_test, ntrain, ntest, 5, best_train, rall)),

    # ("best_train",
    #  sm.get_oof_lgb, (cp.lgb_paras_regressor(7, "regression", 1500), best_train, y_train, x_test, ntrain, ntest, 5)),
    # ("train_1",
    #  sm.get_oof_lgb, (cp.lgb_paras_regressor(7, "regression", 1500), train_1, y_train, x_test, ntrain, ntest, 5)),
    #
    # ("train_4",
    #  sm.get_oof_lgb, (cp.lgb_paras_regressor(7, "regression", 1500), train_4, y_train, x_test, ntrain, ntest, 5)),
    # ("best_train_9",
    #  sm.get_oof_lgb, (cp.lgb_paras_regressor(9, "regression", 1500), best_train, y_train, x_test, ntrain, ntest, 5)),
    # ("train_2_5",
    #  sm.get_oof_lgb, (cp.lgb_paras_regressor(11, "regression", 1500), train_2, y_train, x_test, ntrain, ntest, 5)),
    # ("train_3_5",
    #  sm.get_oof_lgb, (cp.lgb_paras_regressor(8, "regression", 1500), train_3, y_train, x_test, ntrain, ntest, 5)),
    # ("random_5",
    #  sm.get_oof_lgb, (cp.lgb_paras_regressor(5, "regression", 1500), train_5, y_train, x_test, ntrain, ntest, 5)),
    # ("random_7",
    #  sm.get_oof_lgb, (cp.lgb_paras_regressor(7, "regression", 1000), train_5, y_train, x_test, ntrain, ntest, 5)),
    # ("random_9",
    #  sm.get_oof_lgb, (cp.lgb_paras_regressor(9, "regression", 1200), train_5, y_train, x_test, ntrain, ntest, 5)),
    # ("lg13",
    #  sm.get_oof_lgb, (cp.lgb_paras_regressor(8, "fair", 2000), x_train, y_train, x_test, ntrain, ntest, 5)),
    # ("lg14",
    #  sm.get_oof_lgb, (cp.lgb_paras_regressor(8, "huber", 2000), x_train, y_train, x_test, ntrain, ntest, 5)),
    # ("lg15",
    #  sm.get_oof_lgb, (cp.lgb_paras_regressor(8, "huber", 1000), x_train, y_train, x_test, ntrain, ntest, 5)),
    # ("ada", sm.get_oof_regressor, (ada, x_train, y_train, x_test, ntrain, ntest)),
    # ("gb", sm.get_oof_regressor, (gb, x_train, y_train, x_test, ntrain, ntest)),
    # ("et", sm.get_oof_regressor, (et, x_train, y_train, x_test, ntrain, ntest)),
    # ("le", sm.get_oof_regressor, (le, x_train, y_train, x_test, ntrain, ntest)),
    # ("el", sm.get_oof_regressor, (el, x_train, y_train, x_test, ntrain, ntest)),
    # ("rd", sm.get_oof_regressor, (rd, x_train, y_train, x_test, ntrain, ntest)),
    # ("ls", sm.get_oof_regressor, (ls, x_train, y_train, x_test, ntrain, ntest)),
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
