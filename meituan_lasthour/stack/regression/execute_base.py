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

import solutions.meituan_lasthour.lgbm_cv as lc
from sklearn.model_selection import RandomizedSearchCV

SEED = 2018
args = lc.init_arguments()
x_train, x_test, y_train, ntrain, ntest = feg.load_base_data(args.train_path, args.test_path)

y_train = y_train.ravel()

print(x_train.shape)

# ---------------------------------------get_oof_tree(clf, x_train, y_train, x_test, ntrain, ntest, NFOLDS = 5)------------------------------------------------------


rf = RandomForestRegressor(n_estimators = 500, max_depth = 8, n_jobs = -1,  random_state=SEED)
ada = AdaBoostRegressor(n_estimators = 80, learning_rate = 0.004, loss = 'square', random_state=SEED)
gb = GradientBoostingRegressor(learning_rate = 0.01, n_estimators=120, subsample = 0.921, max_depth = 5, random_state=SEED)
et = ExtraTreesRegressor(n_estimators = 150, max_depth= 6,max_features='sqrt',n_jobs = - 1, random_state = SEED)
le = LinearRegression(n_jobs=-1)
el = ElasticNet(random_state=SEED)
rd = Ridge(random_state=SEED)
ls = Lasso( random_state=SEED)

rf = RandomizedSearchCV(rf, rf_parameters, n_iter=30, verbose=10, n_jobs=-1, cv=3, scoring='neg_mean_absolute_error')
ada = RandomizedSearchCV(ada, rf_parameters, n_iter=30, verbose=10, n_jobs=-1, cv=3, scoring='neg_mean_absolute_error')
gb = RandomizedSearchCV(gb, rf_parameters, n_iter=30, verbose=10, n_jobs=-1, cv=3, scoring='neg_mean_absolute_error')
et = RandomizedSearchCV(et, rf_parameters, n_iter=30, verbose=10, n_jobs=-1, cv=3, scoring='neg_mean_absolute_error')

# xgb_reg_train5, xgb_reg_test5 = sm.get_oof_xgb_linear(cp.xgb_params_level_5, x_train, y_train, x_test, ntrain, ntest,
#                                                       NFOLDS=5)
# xgb_reg_train10, xgb_reg_test10 = sm.get_oof_xgb_linear(cp.xgb_params_level_10, x_train,y_train,x_test,ntrain,ntest, NFOLDS = 5)
#
# lgb_reg_train3, lgb_reg_test3 = sm.get_oof_lgb(cp.lgb_params_regressor_3, x_train,y_train,x_test,ntrain,ntest, NFOLDS = 5)
# lgb_reg_train8, lgb_reg_test8 = sm.get_oof_lgb(cp.lgb_params_regressor_8, x_train,y_train,x_test,ntrain,ntest, NFOLDS = 5)

para_set = [
    # sm.get_oof_xgb_linear(cp.xgb_params_level_5, x_train, y_train, x_test, ntrain, ntest, NFOLDS=5),
    # sm.get_oof_xgb_linear(cp.xgb_params_level_8, x_train, y_train, x_test, ntrain, ntest, NFOLDS=5),
    # sm.get_oof_xgb_linear(cp.xgb_params_level_10, x_train, y_train, x_test, ntrain, ntest, NFOLDS=5),
    sm.get_oof_lgb(cp.lgb_paras_regressor(2), x_train, y_train, x_test, ntrain, ntest, NFOLDS=5),
    sm.get_oof_lgb(cp.lgb_paras_regressor(4), x_train, y_train, x_test, ntrain, ntest, NFOLDS=5),
    sm.get_oof_lgb(cp.lgb_paras_regressor(6), x_train, y_train, x_test, ntrain, ntest, NFOLDS=5),
    sm.get_oof_lgb(cp.lgb_paras_regressor(8), x_train, y_train, x_test, ntrain, ntest, NFOLDS=5),
    sm.get_oof_lgb(cp.lgb_paras_regressor(10), x_train, y_train, x_test, ntrain, ntest, NFOLDS=5),
    sm.get_oof_regressor(rf, x_train, y_train, x_test, ntrain, ntest),
    sm.get_oof_regressor(ada, x_train, y_train, x_test, ntrain, ntest),
    sm.get_oof_regressor(gb, x_train, y_train, x_test, ntrain, ntest),
    sm.get_oof_regressor(et, x_train, y_train, x_test, ntrain, ntest),
    sm.get_oof_regressor(le, x_train, y_train, x_test, ntrain, ntest),
    sm.get_oof_regressor(el, x_train, y_train, x_test, ntrain, ntest),
    sm.get_oof_regressor(rd, x_train, y_train, x_test, ntrain, ntest),
    sm.get_oof_regressor(ls, x_train, y_train, x_test, ntrain, ntest),
]

names_one = [
    # 'xgb5',
    # 'xgb8',
    # 'xgb10',
    'lgb2',
    'lgb4',
    'lgb6',
    'lgb8',
    'lgb10',
    'rf',
    'ada',
    'gb',
    'et',
    'le',
    'el',
    'rd',
    'ls'
]

x_train = pd.DataFrame(np.hstack([x[0] for x in para_set]))
x_test = pd.DataFrame(np.hstack([x[1] for x in para_set]))
x_train.columns = names_one
x_test.columns = names_one

x_train.to_csv("stack_data/train" + str(SEED) + ".csv", index=None)
x_test.to_csv("stack_data/test" + str(SEED) + ".csv", index=None)

