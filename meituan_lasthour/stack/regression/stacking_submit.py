import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import r2_score

pd.options.mode.chained_assignment = None

# read datasets

train = pd.read_csv('/Users/dongjian/data/meituanKaggleData/train_sub.csv')
test = pd.read_csv('/Users/dongjian/data/meituanKaggleData/test_sub.csv')
train = train.fillna(0)
test = test.fillna(0)

test_id = test.order_id
y_train = train.delivery_duration

# ----------------read the base model prediction files-------------------
x_train = pd.read_csv(
        "/Users/dongjian/work/meituan/instacart/solutions/meituan_lasthour/stack/regression/stack_data/train2018.csv")  # define yourself
x_test = pd.read_csv(
        "/Users/dongjian/work/meituan/instacart/solutions/meituan_lasthour/stack/regression/stack_data/test2018.csv")

name_col = ["col" + str(i) for i in range(x_train.shape[1])]
x_test.columns = name_col
x_train.columns = name_col

dtrain = xgb.DMatrix(data=x_train, label=y_train, missing=np.nan)

params = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    # 'eval_metric': 'mae',
    'eta': 0.07,
    "min_child_weight":10,
    "n_jobs": 4,
    # 'num_round': 100,
    'colsample_bytree': 0.9,
    'subsample': 0.9,
    'max_depth': 5,
    'nthread': -1,
    'seed': 20171001,
    'silent': 1,
}


def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)


clf = xgb.cv(params, dtrain, num_boost_round=1000, nfold=6, early_stopping_rounds=15, feval=xgb_r2_score, maximize=True,
             show_progress=True)
# #
watchlist = [(dtrain, 'train')]
#
# lgb.cv
best_rounds = np.argmax(clf['test-r2-mean'])
print("---------------------best_rounds : {}-------------------------------".format(best_rounds))
print(clf.iloc[best_rounds])
files_name = clf.iloc[best_rounds]["test-r2-mean"]
print("------train-----------------------------------------")
bst = xgb.train(params, dtrain, best_rounds, watchlist, verbose_eval=10)
print("------predict---------------------------------------")
dtest = xgb.DMatrix(data=x_test)
preds = bst.predict(dtest)
output = pd.DataFrame({'order_id': test_id.astype(np.int64), 'delivery_duration': preds})
print("------file generate---------------------------------")
# output.to_csv('/Users/dongjian/work/meituan/instacart/solutions/meituan_lasthour/stack/regression/upload/cv_' + str(best_rounds) + '_' + str(files_name) + '_' + 'my_preds.csv', index=None)
output.to_csv('/Users/dongjian/work/meituan/instacart/solutions/meituan_lasthour/stack/regression/upload/out3.csv',
              index=None)


