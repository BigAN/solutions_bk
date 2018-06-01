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
           "rain_bin", "wind_bin", "temperature_bin",
           u'high_level_arrive_guest##delivery_distance_3',
           u'high_level_arrive_guest##delivery_distance_4'
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
    'weekday',
    "poi_agg__box_total_value__median_by__poi_id",
    "lasthour_poi_agg__box_total_value__median_by__poi_id",
    "lasthour_poi_agg__waiting_order_num__median_by__poi_id",
    "food_num"

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
    return filter(lambda x: x, map(lambda x: x if x.startswith(tgt) else None, list(to_do)))


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


def sum_real_ae(a, b):
    return sum(a - b) / len(a)


def gene_xtrain(train, tgt_list, remove=None):
    features = list(set(train.columns.tolist()[1:]) - set(lc.to_drop) - set(lc.fea_drop))
    rs = []
    for i in tgt_list:
        if i == "base":
            rs += lc.generate_base()
        else:
            rs += lc.generate_tgt(features, i)

    print "rs,len(rs) {} ".format(len(rs))
    if remove:
        for r in remove:
            rs = filter(lambda x: r not in x, rs)
            print "r {} ,rs,len(rs) {} ".format(r, len(rs))
    # print rs
    return rs


def random_gene_train(train, times):
    features = list(set(train.columns.tolist()[1:]) - set(lc.to_drop) - set(lc.fea_drop))
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

    train_data.fillna(-999, inplace=True)
    test_data.fillna(-999, inplace=True)

    train_data.loc[:, "delivery_duration_bin"] = train_data["delivery_duration"].apply(lambda x: int(x / 1000))
    # train_data = train_data[train_data.delivery_duration_bin>=4]
    test_data.loc[:, "delivery_duration_bin"] = test_data["delivery_duration"].apply(lambda x: int(x / 1000))
    print test_data.groupby("delivery_duration_bin").agg({"order_id": "count"})
    base_train = lc.gene_xtrain(train_data, ["base"])
    best_train = lc.gene_xtrain(train_data,
                                ["base", "poi_agg", "area_id_area_id#hour#minute"])
    train_1 = lc.gene_xtrain(train_data, ["base", "poi_agg", "area_id_area_id#hour#minute", "cur_block"])
    train_2 = lc.gene_xtrain(train_data, ["base", "poi_agg", "area_id_area_id#hour#minute", "cluster"])
    train_3 = lc.gene_xtrain(train_data, ["base", "area_id_area_id#hour#minute", "cluster"])
    train_4 = lc.gene_xtrain(train_data, ["poi_agg", "cluster", "area_id_area_id#hour#minute"])
    # train_5 = random_gene_train(train_data, 80)


    labels = train_data[args.label].values.astype(np.float32).flatten()
    labels_val = test_data[[args.label]].values.astype(np.float32).flatten()
    rs = []
    for n, feas in enumerate([
        ["base", "area_agg", "poi_agg", "lasthour_poi_agg", "user_bin2", "poi_bin2", "poi_user_bin2"],
        # ["base", "area_agg", "area_10m_agg", "poi_agg", "lasthour_poi_agg", "user_bin2", "poi_bin2", "poi_user_bin2"],
    ]):
        train_feas = lc.gene_xtrain(train_data, feas, remove=["max", "min"])
        print "round {},name is {}, train_feas is {} ".format(n, "@".join(feas), train_feas)
        lgb_train = lgb.Dataset(train_data[train_feas], labels)
        lgb_eval = lgb.Dataset(test_data[train_feas], labels_val)

        print('Start training...')

        params = {'num_leaves': 100, 'task': 'train', 'verbose': 1, 'learning_rate': 0.03, 'nthread': 8,
                  'min_data_in_leaf': 10, 'objective': 'regression_l2', 'boosting_type': 'gbdt', 'metric': 'mae',
                  'feature_fraction': 0.7}
        # 300 -1200 2800

        labels3 = train_data[train_data.delivery_duration > 3500][args.label].values.astype(np.float32).flatten()
        lgb_train3 = lgb.Dataset(train_data[train_data.delivery_duration > 3500][train_feas], labels3)
        labels_val3 = test_data[test_data.delivery_duration > 3500][args.label].values.astype(np.float32).flatten()
        lgb_eval3 = lgb.Dataset(test_data[test_data.delivery_duration > 3500][train_feas], labels_val3,
                                reference=lgb_train3)

        labels2 = train_data[(train_data.delivery_duration < 5000) & (train_data.delivery_duration > 600)][
            args.label].values.astype(np.float32).flatten()
        lgb_train2 = lgb.Dataset(
                train_data[(train_data.delivery_duration < 5000) & (train_data.delivery_duration > 600)][train_feas],
                labels2)
        labels_val2 = test_data[(test_data.delivery_duration < 5000) & (test_data.delivery_duration > 600)][
            args.label].values.astype(np.float32).flatten()
        lgb_eval2 = lgb.Dataset(
                test_data[(test_data.delivery_duration < 5000) & (test_data.delivery_duration > 600)][train_feas],
                labels_val2,
                reference=lgb_train2)

        labels1 = train_data[(train_data.delivery_duration < 800) & (train_data.delivery_duration > 300)][
            args.label].values.astype(np.float32).flatten()
        lgb_train1 = lgb.Dataset(
                train_data[(train_data.delivery_duration < 800) & (train_data.delivery_duration > 300)][train_feas],
                labels1)
        labels_val1 = test_data[(test_data.delivery_duration < 800) & (test_data.delivery_duration > 300)][
            args.label].values.astype(np.float32).flatten()
        lgb_eval1 = lgb.Dataset(
                test_data[(test_data.delivery_duration < 800) & (test_data.delivery_duration > 300)][train_feas],
                labels_val1,
                reference=lgb_train1)

        labels = train_data[(train_data.delivery_duration < 5000) & (train_data.delivery_duration > 300)][
            args.label].values.astype(np.float32).flatten()
        lgb_train = lgb.Dataset(
                train_data[(train_data.delivery_duration < 5000) & (train_data.delivery_duration > 300)][train_feas],
                labels)
        labels_val = test_data[(test_data.delivery_duration < 5000) & (test_data.delivery_duration > 300)][
            args.label].values.astype(np.float32).flatten()
        lgb_eval = lgb.Dataset(
                test_data[(test_data.delivery_duration < 5000) & (test_data.delivery_duration > 300)][train_feas],
                labels_val,
                reference=lgb_train)

        gbm1 = lgb.train(params,
                         lgb_train1,
                         # fobj=huber_approx_obj,
                         valid_sets=[lgb_train1, lgb_eval1],
                         num_boost_round=args.round,
                         early_stopping_rounds=100,
                         verbose_eval=100)

        gbm2 = lgb.train(params,
                         lgb_train2,
                         # fobj=huber_approx_obj,
                         valid_sets=[lgb_train2, lgb_eval2],
                         num_boost_round=args.round,
                         early_stopping_rounds=100,
                         verbose_eval=100)

        gbm3 = lgb.train(params,
                         lgb_train3,
                         # fobj=huber_approx_obj,
                         valid_sets=[lgb_train3, lgb_eval3],
                         num_boost_round=args.round,
                         early_stopping_rounds=100,
                         verbose_eval=100)

        gbm = lgb.train(params,
                        lgb_train,
                        # fobj=huber_approx_obj,
                        valid_sets=[lgb_train, lgb_eval],
                        num_boost_round=args.round,
                        early_stopping_rounds=100,
                        verbose_eval=100)

        df = pd.DataFrame({'feature': gbm.feature_name(), 'importances': gbm.feature_importance()})
        print(df.sort_values('importances'))

        test_data.loc[:, "delivery_duration_prd"] = gbm.predict(test_data[train_feas])
        test_data.loc[:, "delivery_duration_prd1"] = gbm1.predict(test_data[train_feas])
        test_data.loc[:, "delivery_duration_prd2"] = gbm2.predict(test_data[train_feas])
        test_data.loc[:, "delivery_duration_prd3"] = gbm3.predict(test_data[train_feas])


        def split_prd(df):
            if df.delivery_duration_prd > 3500:
                return df.delivery_duration_prd3
            elif df.delivery_duration_prd > 800:
                return df.delivery_duration_prd2
            else:
                return df.delivery_duration_prd1


        test_data.loc[:, "delivery_duration_prd"] = test_data \
            .apply(split_prd, axis=1)

        print "!!!!!!!!!!!!!! rs is {} !!!!!!!!!!!!!!!!!".format(
                mean_absolute_error(test_data[args.label], test_data["delivery_duration_prd"]))
        tmp = []
        all = len(test_data) * mean_absolute_error(test_data[args.label], test_data["delivery_duration_prd"])

        for i in [0, 1, 2, 3, 4, 5, 6, 7]:
            print i
            print "!!!!!!!!!!!!!! bin is {} rs is {} !!!!!!!!!!!!!!!!!". \
                format(i,
                       mean_absolute_error(test_data[test_data.delivery_duration_bin == i][args.label],
                                           test_data[test_data.delivery_duration_bin == i]["delivery_duration_prd"]))
            tmp.append({"mae": mean_absolute_error(test_data[test_data.delivery_duration_bin == i][args.label],
                                                   test_data[test_data.delivery_duration_bin == i][
                                                       "delivery_duration_prd"]),
                        "mae_count": len(test_data[test_data.delivery_duration_bin == i]),
                        "mae_real":
                            sum_real_ae(
                                    test_data[test_data.delivery_duration_bin == i][
                                        "delivery_duration_prd"],
                                    test_data[test_data.delivery_duration_bin == i][args.label]),

                        "mae_total": len(test_data[test_data.delivery_duration_bin == i]) * mean_absolute_error(
                                test_data[test_data.delivery_duration_bin == i][args.label],
                                test_data[test_data.delivery_duration_bin == i]["delivery_duration_prd"]),
                        "mae_percent": len(test_data[test_data.delivery_duration_bin == i]) * mean_absolute_error(
                                test_data[test_data.delivery_duration_bin == i][args.label],
                                test_data[test_data.delivery_duration_bin == i]["delivery_duration_prd"]) / all,
                        "mae_prd": 0 if (
                            test_data[(test_data.delivery_duration_prd / (10 ** 3)).astype(int) == i].empty) else
                        mean_absolute_error(
                                test_data[(test_data.delivery_duration_prd / (10 ** 3)).astype(int) == i][args.label],
                                test_data[(test_data.delivery_duration_prd / (10 ** 3)).astype(int) == i][
                                    "delivery_duration_prd"]),
                        "mae_prd_count": 0 if (
                            test_data[(test_data.delivery_duration_prd / (10 ** 3)).astype(int) == i].empty) else
                        len(test_data[(test_data.delivery_duration_prd / (10 ** 3)).astype(int) == i]),
                        "prd_mean": 0 if (
                            test_data[(test_data.delivery_duration_prd / (10 ** 3)).astype(int) == i].empty) else
                        np.mean(test_data[(test_data.delivery_duration_prd / (10 ** 3)).astype(int) == i][args.label]),
                        "mean": 0 if (
                            test_data[(test_data.delivery_duration_prd / (10 ** 3)).astype(int) == i].empty) else
                        np.mean(test_data[test_data.delivery_duration_bin == i][args.label]),
                        "mae_prd_real": 0 if (
                            test_data[(test_data.delivery_duration_prd / (10 ** 3)).astype(int) == i].empty) else
                        sum_real_ae(
                                test_data[(test_data.delivery_duration_prd / (10 ** 3)).astype(int) == i][
                                    "delivery_duration_prd"],
                                test_data[(test_data.delivery_duration_prd / (10 ** 3)).astype(int) == i][args.label])
                        })
        print "prd"
        print test_data["delivery_duration_prd"].describe()
        print "actual"
        print test_data[args.label].describe()
        rs.append([feas, mean_absolute_error(test_data[args.label], test_data["delivery_duration_prd"]), tmp])

    print sorted(rs, key=lambda x: x[1])
    test_data.to_csv(args.out_path)
