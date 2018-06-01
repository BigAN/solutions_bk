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
# import lgbm_cv as lc
import numpy as np

from sklearn.metrics import mean_absolute_error

cate_features = ["poi_id", "area_id", "15min", "10min", "direction", "poi_lat_bin", "poi_lng_bin", "cst_lat_bin",
                 "cst_lng_bin"]

basic_features = ["order_id", "poi_id", "area_id", "food_total_value", "box_total_value", "food_num",
                  "delivery_distance", "order_unix_time", "arriveshop_unix_time", "fetch_unix_time", "finish_unix_time",
                  "customer_longitude", "customer_latitude", "poi_lng", "poi_lat", "waiting_order_num",
                  "delivery_duration", "tmp_order_unix_time", "day", "weekday", "hour", "10min", "high", "weekday_hour",
                  "_10min_not_fetched_order_num", "_10min_working_rider_num", "_10min_notbusy_working_rider_num",
                  "_10min_deliverying_order_num", "_10min_rain", "_10min_temperature", "_10min_wind", ]

advanced_features = [
    "arriveshop_cost", "fetch_cost", "arrive_guest_cost", "avg_speed", "direction", "poi_lat_bin",
    "poi_lng_bin", "cst_lat_bin", "cst_lng_bin", "pickup_cluster", "dropoff_cluster",
    "_avg_speed_hour", "_avg_speed_weekday", "_avg_speed_10min", "_avg_speed_weekday_hour",
    "_avg_speed_pickup_cluster", "_avg_speed_dropoff_cluster", "coor_count_x",
    "_avg_speed_hour_pickup_cluster", "coor_count_y", "_avg_speed_hour_dropoff_cluster", "coor_count",
    "_avg_speed_pickup_cluster_dropoff_cluster",
]

to_drop = ['order_unix_time', 'arriveshop_unix_time', 'fetch_unix_time', 'finish_unix_time', 'order_id',
           'delivery_duration', 'date', "arriveshop_cost", "fetch_cost", "arrive_guest_cost", "tmp_order_unix_time",
           "coor_count_y", "coor_count_x",
           "coor_count",
           "avg_speed", "arrive_guest_avg_speed", 'high', 'day', "coor_count_x.1", "coor_count_y.1", "coor_count_y.2",
           'coor_count_x.2',
           "rain_bin", "wind_bin", "temperature_bin"
           ]

fea_drop = [
    "_10min_wind",
    # "area_id",
    "poi_id_mean_box_total_value",
    # "box_total_value",
    # "food_num",
    "_10min_rain",
    "weekday_hour",
    "_avg_speed_weekday",
    "minute",
    "_10min_temperature",
    "hour",
    "pickup_cluster",
    "dropoff_cluster",
    "next15_min",
    "cur_min_x",
    "cur_min_y",
    "cur_min",
    "next5_min",
    "next10_min",
    "next15_min",
    "cur_10_block",
    "cur_10_block_y.2",
    "cur_10_block_y.1",
    "cur_10_block_y",
    "cur_10_block_x",
    "cur_10_block_x.2",
    "cur_10_block_x.1",
    "last_10_block",
    "next_10_block",
    "area_id_y",
    "area_id_x",
    'weekday'

    # "dropoff_cluster",
    # "delivery_distance",
    # "_10min_deliverying_order_num"
    # "poi_lat_bin","poi_lng_bin","cst_lat_bin","cst_lng_bin"
    # "poi_id",
    # "weekday",
    # "food_num",
    # "lasthour_poi_agg_poi_id_mean_delivery_distance"
]  # 407.263830991 all 400


# features = list((set(basic_features) | set(advanced_features)) - set(to_drop) - set(fea_drop))


# features = single_feas


def generate_tgt(inp_features, tgt):
    to_do = set(inp_features) - set(to_drop) - set(fea_drop)
    return filter(lambda x: x, map(lambda x: x if tgt in x else None, list(to_do)))


def generate_base():
    return list(
            set(["order_id", "poi_id", "area_id", "food_total_value", "box_total_value", "food_num",
                 "delivery_distance", "order_unix_time", "arriveshop_unix_time", "fetch_unix_time", "finish_unix_time",
                 "customer_longitude", "customer_latitude", "poi_lng", "poi_lat", "waiting_order_num",
                 "delivery_duration", "tmp_order_unix_time", "day", "weekday", "hour", "10min", "15min", "cur_min",
                 "next5_min", "next10_min", "cur_10_block", "last_10_block", "next_10_block", "minute", "high",
                 "weekday_hour", "_10min_not_fetched_order_num", "_10min_working_rider_num",
                 "_10min_notbusy_working_rider_num", "_10min_deliverying_order_num", "_10min_rain",
                 "_10min_temperature", "_10min_wind", "area_busy_coef", "bill_number_per_rider", "avg_speed",
                 "direction", "poi_lat_bin", "poi_lng_bin", "cst_lat_bin", "cst_lng_bin", "arriveshop_cost",
                 "fetch_cost", "arrive_guest_cost", "arrive_guest_avg_speed", "pickup_cluster",
                 "dropoff_cluster", "prd_arrive_guest_time"]) - set(to_drop) - set(fea_drop))


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


import random


def gene_xtrain(train, tgt_list):
    features = list(set(train.columns.tolist()[1:]) - set(to_drop) - set(fea_drop))
    rs = []
    for i in tgt_list:
        if i == "base":
            rs += generate_base()
        else:
            rs += generate_tgt(features, i)
    print rs
    return rs


def random_gene_train(train, times):
    features = list(set(train.columns.tolist()[1:]) - set(to_drop) - set(fea_drop))
    rs = []
    for n in xrange(times):
        rs += [random.choice(features)]
    return list(set(rs))


if __name__ == '__main__':
    args = init_arguments()

    train_data = pd.read_csv(args.train_path)
    test_data = pd.read_csv(args.test_path)
    train_data.head(5)
    features = list(set(train_data.columns.tolist()[1:]) - set(to_drop) - set(fea_drop))

    print "features,length {}, {}".format(len(features), features)

    train_data.fillna(0, inplace=True)
    test_data.fillna(0, inplace=True)

    train_data.loc[:, "delivery_duration_bin"] = train_data["delivery_duration"].apply(lambda x: int(x / 1000))
    # train_data = train_data[train_data.delivery_duration_bin>=4]
    test_data.loc[:, "delivery_duration_bin"] = test_data["delivery_duration"].apply(lambda x: int(x / 1000))
    print test_data.groupby("delivery_duration_bin").agg({"order_id": "count"})
    base_train = gene_xtrain(train_data, ["base"])
    best_train = gene_xtrain(train_data,
                                ["base", "poi_agg", "area_id_area_id#hour#minute"])
    train_1 = gene_xtrain(train_data, ["base", "poi_agg", "area_id_area_id#hour#minute", "cur_block"])
    train_2 = gene_xtrain(train_data, ["base", "poi_agg", "area_id_area_id#hour#minute", "cluster"])
    train_3 = gene_xtrain(train_data, ["base", "area_id_area_id#hour#minute", "cluster"])
    train_4 = gene_xtrain(train_data, ["poi_agg", "cluster", "area_id_area_id#hour#minute"])
    train_5 = random_gene_train(train_data, 80)


    labels = train_data[args.label].values.astype(np.float32).flatten()
    labels_val = test_data[[args.label]].values.astype(np.float32).flatten()

    for n, train_feas in enumerate([
        # base_train,
        best_train,
        # train_1, train_2, train_3, train_4, train_5
    ]):
        print "round {}, train_feas is {} ".format(n, train_feas)
        lgb_train = lgb.Dataset(train_data[train_feas], labels)
        lgb_eval = lgb.Dataset(test_data[train_feas], labels_val, reference=lgb_train)

        print('Start training...')

        params = {'num_leaves': 100, 'task': 'train', 'verbose': 1, 'learning_rate': 0.01, 'nthread': 4,
                  'min_data_in_leaf': 10, 'objective': 'regression_l2', 'boosting_type': 'gbdt', 'metric': 'mae',
                  'feature_fraction': 0.7}

        gbm = lgb.train(params,
                        lgb_train,
                        # fobj=huber_approx_obj,
                        valid_sets=[lgb_train, lgb_eval],
                        num_boost_round=args.round,
                        early_stopping_rounds=100,
                        verbose_eval=100)
        print gbm.best_score
        print('Feature names:', gbm.feature_name())

        print('Calculate feature importances...')
        # feature importances
        print('Feature importances:', list(gbm.feature_importance()))

        df = pd.DataFrame({'feature': gbm.feature_name(), 'importances': gbm.feature_importance()})
        print(df.sort_values('importances'))

        test_data.loc[:, "delivery_duration_prd"] = gbm.predict(test_data[train_feas])

        print "!!!!!!!!!!!!!! rs is {} !!!!!!!!!!!!!!!!!".format(
                mean_absolute_error(test_data[args.label], test_data["delivery_duration_prd"]))

        for i in [0, 1, 2, 3, 4, 5]:
            print i
            print "!!!!!!!!!!!!!! bin is {} rs is {} !!!!!!!!!!!!!!!!!". \
                format(i,
                       mean_absolute_error(test_data[test_data.delivery_duration_bin == i][args.label],
                                           test_data[test_data.delivery_duration_bin == i]["delivery_duration_prd"]))
        print "prd"
        print test_data["delivery_duration_prd"].describe()
        print "actual"
        print test_data[args.label].describe()

    test_data.to_csv(args.out_path)
