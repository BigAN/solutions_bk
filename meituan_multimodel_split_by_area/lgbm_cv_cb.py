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
import catboost as cb
from sklearn.metrics import mean_absolute_error
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
    "_avg_speed_pickup_cluster_dropoff_cluster"
]

to_drop = ['order_unix_time', 'arriveshop_unix_time', 'fetch_unix_time', 'finish_unix_time', 'order_id',
           'delivery_duration', 'date', "arriveshop_cost", "fetch_cost", "arrive_guest_cost", "tmp_order_unix_time",
           "_poi_min_dd", "_poi_mean_dd", "_poi_max_dd",
           "_poi_mean_delivery_distance", "area_id", "_area_median_not_fetched_order_num",
           "_area_median_working_rider_num", "_area_median_notbusy_working_rider_num",
           "_area_median_deliverying_order_num",
           "poi_lat_bin", "coor_count_y", "coor_count_x",
           "poi_lng_bin", "cst_lat_bin", "cst_lng_bin", "coor_count",
           "avg_speed", 'day', 'high'
           ]

fea_drop = [
    # "temperature", "poi_lng", "rain", "poi_id", "working_rider_num", "deliverying_order_num", "food_num",
    # "minute", "customer_latitude"
]  # 407.263830991 all 400

features = list((set(basic_features) | set(advanced_features)) - set(to_drop) - set(fea_drop))

def init_arguments():
    parser = argparse.ArgumentParser()

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
    data = pd.read_csv(args.train_path)
    train_data = pd.read_csv(args.train_path)
    test_data = pd.read_csv(args.test_path)
    train_data.head(5)

    train_data.fillna(0, inplace=True)
    test_data.fillna(0, inplace=True)

    order_id = pd.DataFrame(np.unique(test_data.order_id), columns=["order_id"])

    labels = train_data[['delivery_duration']].values.astype(np.float32).flatten()
    labels_val = test_data[['delivery_duration']].values.astype(np.float32).flatten()

    lgb_train = lgb.Dataset(train_data[features], labels)
    lgb_eval = lgb.Dataset(test_data[features], labels_val, reference=lgb_train)

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

    print('Start training...')
    # train
    cat = cb.CatBoostRegressor(iterations=args.round,
                         learning_rate=0.03,
                         depth=8,
                         l2_leaf_reg=3,
                         rsm=1,
                         loss_function='Logloss' ).fit(train_data[features],labels)

    test_data.loc[:, "delivery_duration_prd"] = cat.predict(test_data[features])

    print mean_absolute_error(test_data["delivery_duration"], test_data["delivery_duration_prd"])
    # rs = order_id.merge(test_data[["order_id", "delivery_duration"]], left_on="order_id", right_on="order_id",
    #                     how="left")
    #
    # rs.to_csv(args.out_path, header=['order_id', 'delivery_duration'], index=False)
