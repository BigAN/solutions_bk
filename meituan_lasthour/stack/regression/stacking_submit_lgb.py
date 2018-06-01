import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import r2_score
import solutions.meituan_lasthour.lgbm_cv as lc
pd.options.mode.chained_assignment = None

# read datasets
args = lc.init_arguments()

train = pd.read_csv(args.train_path)
test = pd.read_csv(args.test_path)
# train = train.fillna(0)
# test = test.fillna(0)

test_id = test.order_id
y_train = train.delivery_duration

# ----------------read the base model prediction files-------------------
x_train = pd.read_csv(
        "/Users/dongjian/work/meituan/instacart/solutions/meituan_lasthour/stack/regression/stack_data/train2018.csv")  # define yourself
x_test = pd.read_csv(
        "/Users/dongjian/work/meituan/instacart/solutions/meituan_lasthour/stack/regression/stack_data/test2018.csv")
name_col = list(set(x_train.columns.tolist()))

# name_col = ["col" + str(i) for i in range(x_train.shape[1])]
x_test.columns = name_col
x_train.columns = name_col

dtrain = lgb.Dataset(x_train, y_train)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'mae'},
    'num_leaves': 200,
    'min_sum_hessian_in_leaf': 10,
    "min_data_in_leaf":10,
    'max_depth': -12,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'verbose': 1
}


def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds),True


clf = lgb.cv(params, dtrain, num_boost_round=10**5, nfold=5, early_stopping_rounds=15,feval=xgb_r2_score,
             verbose_eval=10)


best_rounds = np.argmax(clf['r2-mean'])
print("---------------------best_rounds : {}-------------------------------".format(best_rounds))
# print(clf.iloc[best_rounds])
# files_name = clf.iloc[best_rounds]["l1-mean"]
print("------train-----------------------------------------")
gbm = lgb.train(params, dtrain, best_rounds, valid_sets=dtrain, verbose_eval=10,early_stopping_rounds=20)
print("------predict---------------------------------------")
# dtest = lgb.Dataset(data=x_test)
preds = gbm.predict(x_test)
output = pd.DataFrame({'order_id': test_id.astype(np.int64), 'delivery_duration': preds})
print("------file generate---------------------------------")
# output.to_csv('/Users/dongjian/work/meituan/instacart/solutions/meituan_lasthour/stack/regression/upload/cv_' + str(best_rounds) + '_' + str(files_name) + '_' + 'my_preds.csv', index=None)
output.to_csv('/Users/dongjian/work/meituan/instacart/solutions/meituan_lasthour/stack/regression/upload/out3.csv',
              index=None)

print('Feature importances:', list(gbm.feature_importance()))

df = pd.DataFrame({'feature': gbm.feature_name(), 'importances': gbm.feature_importance()})
print(df.sort_values('importances'))
