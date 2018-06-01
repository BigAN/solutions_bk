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
import lightgbm as lgb
import lgbm_cv as lc
from sklearn.metrics import mean_absolute_error
import cv_data_producer as cv

# features = list((set(basic_features) | set(advanced_features)) - set(to_drop) - set(fea_drop))


# features = single_feas
to_tran = [u'area_busy_coef'
    , u'bill_number_per_rider',
           u'area_id_area_id#hour#minute_arriveshop_cost_std',
           u'_10min_not_fetched_order_num',
           u'area_id_area_id#hour#minute_arriveshop_cost_mean',
           u'cur_area_busy_coef', u'poi_agg_poi_id_mean_dd',
           u'_10min_deliverying_order_num', u'cur_10min_not_fetched_order_num_y',
           u'cur_10min_not_fetched_order_num_x', u'future_food_total_value_sum_x',
           u'future_food_total_value_sum_y', u'lasthour_poi_agg_poi_id_mean_dd',
           u'poi_agg_poi_id_std_arriveshop_cost', u'poi_agg_poi_id_std_dd',
           u'area_id_area_id#hour#minute_food_total_value_sum',
           u'waiting_order_num'
           ]


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


if __name__ == '__main__':
    args = init_arguments()

    train_data = pd.read_csv(args.train_path)
    train_data = train_data[cv.gene_mask(train_data, 11)]

    test_data = pd.read_csv(args.test_path)
    train_data.head(5)
    features = list(set(train_data.columns.tolist()[1:]) - set(lc.to_drop) - set(lc.fea_drop))
    print "features,length {}, {}".format(len(features), features)
    test_features = list(set(test_data.columns.tolist()[1:]) - set(lc.to_drop) - set(lc.fea_drop))
    print test_features
    for f in to_tran:
        train_data[f] = train_data[f].apply(lambda x: (1 + x) ** 0.25)
        test_data[f] = test_data[f].apply(lambda x: (1 + x) ** 0.25)
    # train_data.fillna(0, inplace=True)
    # test_data.fillna(0, inplace=True)

    order_id = pd.DataFrame(np.unique(test_data.order_id), columns=["order_id"])
    # import np

    labels = train_data[args.label].values.astype(np.float32).flatten()
    # labels_val = test_data[[args.label]].values.astype(np.float32).flatten()

    lgb_train = lgb.Dataset(train_data[features], labels)
    # lgb_eval = lgb.Dataset(test_data[features], labels_val, reference=lgb_train)

    # for x in ["poi_id","area_id","weekday","10min","15min"]:
    #     train_data[x] = train_data[x].astype("str")

    print('Start training...')

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': "regression",
        'metric': {"mae"},
        'num_leaves': 256,
        'min_sum_hessian_in_leaf': 10,
        'max_depth': -12,
        'learning_rate': 0.03,
        'feature_fraction': 0.6,
        'verbose': 1,
    }


    def huber_approx_obj(preds, dtrain):
        d = preds - dtrain.get_label()  # remove .get_labels() for sklearn
        h = 2000  # h is delta in the graphic
        scale = 1 + (d / h) ** 2
        scale_sqrt = np.sqrt(scale)
        grad = d / scale_sqrt
        hess = 1 / scale / scale_sqrt
        return grad, hess


    def fair_obj(self, preds, dtrain):
        x = preds - dtrain.get_label()
        c = 2

        den = np.abs(x) * np.exp(self.fair_decay * self.iter) + c

        grad = c * x / den
        hess = c * c / den ** 2

        self.iter += 1

        return grad, hess


    # gbm = lgb.train(params,
    #                 lgb_train,
    #                 fobj=huber_approx_obj,
    #                 valid_sets=lgb_eval,
    #                 num_boost_round=args.round,
    #                 verbose_eval=True)
    #


    rs = lgb.cv(params, lgb_train, num_boost_round=args.round, nfold=5, metrics={'mae'}, early_stopping_rounds=20,
                verbose_eval=True,
                # categorical_feature=["area_id","poi_id","weekday","10min","15min"]
                )
    print "rs['l1-mean'][-1],len(rs['l1-mean'])", rs["l1-mean"][-1], len(rs["l1-mean"])





    # print('Feature names:', gbm.feature_name())
    #
    # print('Calculate feature importances...')
    # # feature importances
    # print('Feature importances:', list(gbm.feature_importance()))
    #
    # df = pd.DataFrame({'feature': gbm.feature_name(), 'importances': gbm.feature_importance()})
    # print(df.sort_values('importances'))
    #
    # test_data.loc[:, "delivery_duration_prd"] = gbm.predict(test_data[features])
    #
    # print "!!!!!!!!!!!!!! rs is {} !!!!!!!!!!!!!!!!!".format(
    #         mean_absolute_error(test_data[args.label], test_data["delivery_duration_prd"]))
    # print test_data["delivery_duration_prd"].describe()
    # print test_data[args.label].describe()
    # print "prd"
    # print test_data["delivery_duration_prd"].describe()
    # print "actual"
    # print test_data[args.label].describe()
    #
    # test_data['mae'] = test_data[args.label] - test_data["delivery_duration_prd"]
    #
    # test_data.to_csv("/Users/dongjian/data/meituanKaggleData/test_out.csv")
    # rs = order_id.merge(test_data[["order_id", "delivery_duration"]], left_on="order_id", right_on="order_id",
    #                     how="left")
    #
    # rs.to_csv(args.out_path, header=['order_id', 'delivery_duration'], index=False)
