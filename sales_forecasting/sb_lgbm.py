# encoding:utf8
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import gc
import time
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import argparse
from utils import nice_method as nm
import pandas as pd
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
import datetime
import arrow as ar
from sklearn.model_selection import train_test_split
# print(check_output(["ls", "./data"]).decode("utf8"))


# Any results you write to the current directory are saved as output.


from subprocess import check_output
import datetime
import arrow as ar
import pandas as pd
import numpy as np
import ori_data_reader as dr
import lightgbm as lgb

# print(check_output(["ls", "./data"]).decode("utf8"))

prefix_sep = "#"
features = ['item_nbr',
            'store_nbr',
            'onpromotion']


# Any results you write to the current directory are saved as output.
# input_path = "/Users/dongjian/data/meituanKaggleData/"
# output_path = "/Users/dongjian/data/meituanKaggleData/"

def init_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, dest='input_path',
                        help="data")
    parser.add_argument('--output', type=str, dest='output_path',
                        help="data")
    parser.add_argument('--round', type=int, dest='round',
                        help="data")
    parser.add_argument('--label', type=str, dest='label',
                        help="data")
    return parser.parse_args()


if __name__ == '__main__':
    args = init_arguments()
    data = dr.Data(data_folder=args.input_path, test_the_script=False)

    train_data = data.train
    test_data = data.test

    # features = list(set(train_data.columns.tolist()[1:]) - set(lc.to_drop) - set(lc.fea_drop))
    print "features,length {}, {}".format(len(features), features)
    train_data.head(5)

    # order_id = pd.DataFrame(np.unique(test_data.order_id), columns=["order_id"])

    labels = train_data[[args.label]].values.astype(np.float32).flatten()

    lgb_train = lgb.Dataset(train_data[features], labels)
    # lgb_eval = lgb.Dataset(lgb_train[features], labels, reference=lgb_train)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'RMSE',"l2"},
        'num_leaves': 2**5-1,
        'min_sum_hessian_in_leaf': 10,
        'max_depth': -50,
        'learning_rate': 0.03,
        'feature_fraction': 0.6,
        # 'bagging_fraction': 0.9,
        # 'bagging_freq': 3,
        'verbose': 1
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=[lgb_train],
                    num_boost_round=args.round,
                    early_stopping_rounds=20,
                    verbose_eval=10)

    # print('Feature names:', gbm.feature_name())

    print('Calculate feature importances...')
    # feature importances
    print('Feature importances:', list(gbm.feature_importance()))

    df = pd.DataFrame({'feature': gbm.feature_name(), 'importances': gbm.feature_importance()})
    print(df.sort_values('importances'))

    test_data.loc[:, args.label] = gbm.predict(test_data[features])
    test_data[['id', args.label]].to_csv(args.output_path, header=['id', args.label], index=False)

import h2o
from h2o.automl import H2OAutoML
h2o.init()