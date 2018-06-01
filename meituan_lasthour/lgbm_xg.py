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
           "poi_lat_bin", "coor_count_y", "coor_count_x",
           "poi_lng_bin", "cst_lat_bin", "cst_lng_bin", "coor_count",
           "avg_speed", 'day', 'high', "coor_count_x.1", "coor_count_y.1", "coor_count_y.2", 'coor_count_x.2'
           ]

fea_drop = [
    "pickup_cluster#10min_lasthour_mean_delivery_distance",
    "pickup_cluster#dropoff_cluster#hour#10min_waiting_order_num",
    "pickup_cluster#dropoff_cluster#hour#10min_mean_arrive_guest_cost", "hour#poi_id_mean_delivery_distance",
    "_10min_wind", "area_id", "pickup_cluster#dropoff_cluster#hour_mean_dd",
    "pickup_cluster#10min_lasthour_mean_arriveshop_cost", "poi_id_mean_box_total_value",
    "pickup_cluster#hour_mean_box_total_value", "box_total_value", "10min#poi_id_mean_box_total_value",
    "pickup_cluster#hour_max_dd", "pickup_cluster#dropoff_cluster_mean_box_total_value", "food_num",
    "dropoff_cluster#10min_lasthour_mean_delivery_distance", "dropoff_cluster#10min_lasthour_mean_arrive_guest_cost",
    "poi_id_lasthour_mean_fetch_cost", "pickup_cluster#dropoff_cluster_lasthour_max_dd", "_10min_rain",
    "dropoff_cluster#10min_lasthour_max_dd", "poi_id_mean_food_num", "dropoff_cluster#10min_mean_food_total_value",
    "weekday_hour", "dropoff_cluster#hour_mean_food_total_value", "_avg_speed_weekday",
    "pickup_cluster#dropoff_cluster#hour#10min_min_dd", "10min#poi_id_lasthour_max_dd", "dropoff_cluster#hour_max_dd",
    "pickup_cluster#hour_mean_fetch_cost", "dropoff_cluster#10min_lasthour_mean_arriveshop_cost", "minute",
    "pickup_cluster#dropoff_cluster_lasthour_min_dd", "_10min_temperature",
    "pickup_cluster#10min_lasthour_mean_fetch_cost", "hour"
    #  "dropoff_cluster", "food_num", "poi_lat", "_10min_rain", "weekday_hour",
    # "_avg_speed_pickup_cluster_dropoff_cluster", "weekday", "_avg_speed_10min", "hour"
    # "temperature", "poi_lng", "rain", "poi_id", "working_rider_num", "deliverying_order_num", "food_num",
    # "minute", "customer_latitude"
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

    labels = train_data[args.label].values.astype(np.float32).flatten()
    labels_val = test_data[[args.label]].values.astype(np.float32).flatten()

    dtrain = xgb.DMatrix(train_data[features].values, labels, missing=np.nan)
    dtest = xgb.DMatrix(test_data[features].values, labels_val, missing=np.nan)

    print('training model...')
    watchlist = [(dtrain, 'train')]
    param = {
        'booster': 'gbtree',
        'objective': 'reg:linear',
        # 'eval_metric': 'mae',
        'eta': 0.15,
        'num_round': args.round,
        'colsample_bytree': 0.65,
        'subsample': 0.8,
        'max_depth': 5,
        'nthread': -1,
        'seed': 20171001,
        'silent': 1,
    }
    #
    #
    # def huber_approx_obj(preds, dtrain):
    #     d = preds - dtrain.get_label()
    #     h = 20
    #
    #     scale = 1 + (d / h) ** 2
    #     scale_sqrt = np.sqrt(scale)
    #
    #     grad = d / scale_sqrt
    #     hess = 1 / scale / scale_sqrt
    #
    #     return grad, hess
    #
    #
    # # train
    # def log_cosh_obj(preds, dtrain):
    #     # lgb_train.get_label()
    #     x = preds - dtrain.get_label()
    #     grad = np.tanh(x)
    #     hess = 1 - (grad * grad)
    #     return grad, hess


    bst = xgb.train(param, dtrain, param['num_round'], watchlist, verbose_eval=10)

    print('generating prediction...')
    pred = bst.predict(dtest)

    test_data.loc[:, "delivery_duration_prd"] = pred

    print mean_absolute_error(test_data[args.label], test_data["delivery_duration_prd"])
    print test_data["delivery_duration_prd"].describe()
    print test_data[args.label].describe()


    # rs = order_id.merge(test_data[["order_id", "delivery_duration"]], left_on="order_id", right_on="order_id",
    #                     how="left")
    #
    # rs.to_csv(args.out_path, header=['order_id', 'delivery_duration'], index=False)
