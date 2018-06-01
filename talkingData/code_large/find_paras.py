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
# import sklearn.metrics
# from sklearn.metrics import f1_score, roc_auc_score
# from sklearn.model_selection import train_test_split
import xgboost as xgb
import lgbm_cv as lc
from sklearn.metrics import mean_absolute_error
# from hyperopt import fmin
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


# features = list((set(basic_features) | set(advanced_features)) - set(to_drop) - set(fea_drop))


# features = single_feas

def init_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--label', type=str, dest='label',
                        help="data")

    parser.add_argument('--train', type=str, dest='train_path',
                        help="data")
    parser.add_argument('--test', type=str, dest='test_path',
                        help="data")
    parser.add_argument('--round', type=int, dest='round',
                        help="data", default=500)

    parser.add_argument('--output', type=str, dest='out_path',
                        help="data")
    return parser.parse_args()


log = []
if __name__ == '__main__':
    args = init_arguments()

    train_data = lc.deco_data(pd.read_csv(args.train_path, sep="\t", skiprows=1))
    test_data = lc.deco_data(pd.read_csv(args.test_path, sep="\t", skiprows=1))
    features = list(set(train_data.columns.tolist()) - set(lc.to_drop) - set(lc.fea_drop))
    print "features,length {}, {}".format(len(features), features)

    labels = train_data[args.label].values.astype(np.float32).flatten()
    labels_val = test_data[[args.label]].values.astype(np.float32).flatten()

    dtrain = xgb.DMatrix(train_data[features].values, labels, missing=np.nan)
    dtest = xgb.DMatrix(test_data[features].values, labels_val, missing=np.nan)

    print('training model...')
    best = 0


    def score(params):
        print("Training with params: ")
        print(params)
        num_round = int(params['n_estimators'])
        del params['n_estimators']
        watchlist = [(dtest, 'eval')]
        gbm = xgb.train(params, dtrain, num_round, watchlist, verbose_eval=10, early_stopping_rounds=30)

        score = gbm.best_score
        # TODO: Add the importance for the selected features
        print("\tScore {0}\n\n".format(score))
        log.append([score, gbm.best_iteration, params])
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
            'n_estimators': 200,
            'eta': hp.quniform('eta', 0.025, 0.3, 0.025),
            'eval_metric': 'auc',
            # A problem with max_depth casted to float instead of int with
            # the hp.quniform method.
            'max_depth': hp.choice('max_depth', np.arange(3, 14, dtype=int)),
            'min_child_weight': hp.quniform('min_child_weight', 5, 200, 5),
            'gamma': hp.quniform('gamma', 0, 1, 0.05),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.05, 0.7, 0.05),
            "colsample_bylevel": hp.quniform('colsample_bylevel', 0.05, 0.7, 0.05),
            'subsample': hp.quniform('subsample', 0.3, 1, 0.05),
            "max_delta_step": hp.quniform('max_delta_step', 0.1, 0.7, 0.05),
            'objective': 'binary:logistic',
            'nthread': 8,
            'booster': 'gbtree',
            'silent': 1,
            'seed': random_state
        }
        # Use the fmin function from Hyperopt to find the best hyperparameters
        best = fmin(score, space, algo=tpe.suggest, verbose=True,
                    # trials=trials,
                    # catch_eval_exceptions=True,
                    max_evals=300)
        return best


    best_hyperparams = optimize(
            # trials
    )
    print("The best hyperparameters are: ", "\n")
    print(best_hyperparams)


    # rs = order_id.merge(test_data[["order_id", "delivery_duration"]], left_on="order_id", right_on="order_id",
    #                     how="left")
    #
    # rs.to_csv(args.out_path, header=['order_id', 'delivery_duration'], index=False)
