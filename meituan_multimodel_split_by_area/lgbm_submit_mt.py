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
import cv_data_producer as cdp
import os

to_drop = ['order_unix_time', 'arriveshop_unix_time', 'fetch_unix_time', 'finish_unix_time', 'order_id',
           'delivery_duration', 'date', "arriveshop_cost", "fetch_cost", "arrive_guest_cost", "tmp_order_unix_time",
           "coor_count_y", "coor_count_x",
           "coor_count",
           "avg_speed", "arrive_guest_avg_speed", 'high', 'day', "coor_count_x.1", "coor_count_y.1", "coor_count_y.2",
           'coor_count_x.2',
           "rain_bin", "wind_bin", "temperature_bin","delivery_distance_k"
           ]

fea_drop = [
    "_10min_wind",
    # "area_id",
    "poi_id_mean_box_total_value",
    # "box_total_value",
    # "food_num",
    "poi_id_lasthour_mean_fetch_cost", "_10min_rain",
    "poi_id_mean_food_num",
    "weekday_hour", "_avg_speed_weekday",
    "minute",
    "_10min_temperature",
    "hour"
]  # 407.263830991 all 400

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'mae'},
    'num_leaves': 256,
    'min_sum_hessian_in_leaf': 20,
    'max_depth': -12,
    'learning_rate': 0.05,
    'feature_fraction': 0.6,
    # 'bagging_fraction': 0.9,
    # 'bagging_freq': 3,
    'verbose': 1
}

input_path = '/Users/dongjian/data/meituanKaggleData/'


class area_model(object):
    def __init__(self, id):
        self.id = id
        self.model = None;
        self.train_data = None;
        self.test_data = None;
        self.out_data = None;
        self.features = None;

    def fill_data(self, input_path):
        self.train_data = pd.read_csv(os.path.join(input_path, "train_cv_area_{}.csv".format(str(self.id))))
        self.test_data = pd.read_csv(os.path.join(input_path, "test_cv_area_{}.csv".format(str(self.id))))
        self.features = list(set(self.train_data.columns.tolist()[1:]) - set(to_drop) - set(fea_drop))
        print "features,length {}, {}".format(len(self.features), self.features)

    def train(self):
        labels = self.train_data[args.label].values.astype(np.float32).flatten()

        lgb_train = lgb.Dataset(self.train_data[self.features], labels)

        print('Start training...')
        # train

        self.model = lgb.train(params,
                               lgb_train,
                               num_boost_round=args.round,
                               )

        print('Feature names:', self.model.feature_name())

        print('Calculate feature importances...')
        # feature importances
        print('Feature importances:', list(self.model.feature_importance()))

        df = pd.DataFrame({'feature': self.model.feature_name(), 'importances': self.model.feature_importance()})
        print(df.sort_values('importances'))

    def predict(self):
        self.test_data.loc[:, "delivery_duration_prd"] = self.model.predict(self.test_data[self.features])
        print self.test_data["delivery_duration_prd"].describe()
        self.out_data = self.test_data


def agg_out(area_model_list):
    out = pd.concat(map(lambda x: x.out_data, area_model_list))
    return out.sort_values("order_id")





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


if __name__ == '__main__':
    args = init_arguments()


    def one_model(area_id):
        model = area_model(str(area_id))
        model.fill_data(input_path)
        model.train()
        model.predict()
        return model


    model_list = map(one_model,
                     # [1002435]
                     # cdp.area_id_list
                     [0,1,2]
                     )
    out = agg_out(model_list)
    # print "!!!!!!!!!!!!!! final rs is {} !!!!!!!!!!!!!!!!!".format(
    #         mean_absolute_error(out[args.label], out["delivery_duration_prd"]))

    out[['order_id', 'delivery_duration_prd']].to_csv("/Users/dongjian/data/meituanKaggleData/test_out.csv", header=['order_id', 'delivery_duration'], index=False)
