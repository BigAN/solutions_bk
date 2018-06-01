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

single_feas = ["poi_id", "area_id", "food_total_value", "box_total_value", "food_num",
               "delivery_distance"]

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
    'cur_waiting_order_num',
    'lasthour_poi_agg_poi_id_mean_fetch_cost',
    'poi_agg__avg_speed_stdpoi_id',
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
    test_data = pd.read_csv(args.test_path)
    train_data.head(5)
    features = list(set(train_data.columns.tolist()[1:]) - set(lc.to_drop) - set(lc.fea_drop))
    print "features,length {}, {}".format(len(features), features)

    train_data.fillna(-999, inplace=True)
    test_data.fillna(-999, inplace=True)

    order_id = pd.DataFrame(np.unique(test_data.order_id), columns=["order_id"])

    labels = train_data[args.label].values.astype(np.float32).flatten()
    labels_val = test_data[[args.label]].values.astype(np.float32).flatten()

    lgb_train = lgb.Dataset(train_data[features], labels)
    lgb_eval = lgb.Dataset(test_data[features], labels_val, reference=lgb_train)
    # FOREVER_COMPUTING_FLAG = False

    xgb_pars = []

    for min_data_in_leaf in [1, 10, 30, 50, 100]:
        for learning_rate in [0.01, 0.03, 0.05, 0.1, 0.15]:
            for feature_fraction in [0.7, 0.8, 0.9]:
                for num_leaves in [100, 300, 500, 750, 1000]:
                    for objective in ["regression"]:
                        # for lambda_l2 in [0.5, 1., 1.5, 2., 3.]:
                        for max_bin in [100, 255, 500, 750, 1000]:
                            xgb_pars.append(
                                    {"task": "train",
                                     'boosting_type': 'gbdt',
                                     "min_data_in_leaf": min_data_in_leaf, 'learning_rate': learning_rate,
                                     'feature_fraction': feature_fraction, 'num_leaves': num_leaves,
                                     "objective": objective, 'nthread': 4,
                                     'metric': 'mae',
                                     "max_bin": max_bin,
                                     'verbose': 1
                                        ,})
    rs = []


    # while FOREVER_COMPUTING_FLAG:
    def go(xgb_par):
        # xgb_par = np.random.choice(xgb_pars, 1)[0]

        gbm = lgb.train(params=xgb_par,
                        train_set=lgb_train,
                        num_boost_round=args.round,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=100,
                        verbose_eval=100
                        )
        print(xgb_par)
        print('Modeling mae %.6f' % gbm.best_score["valid_0"]['l1'])
        return xgb_par, (gbm.best_iteration, gbm.best_score["valid_0"]['l1'])


    import multiprocessing as mp

    pool = mp.Pool(4)
    rs = pool.map(go, xgb_pars)
    print sorted(rs, key=lambda x: x[1][1])


    # rs = order_id.merge(test_data[["order_id", "delivery_duration"]], left_on="order_id", right_on="order_id",
    #                     how="left")
    #
    # rs.to_csv(args.out_path, header=['order_id', 'delivery_duration'], index=False)
