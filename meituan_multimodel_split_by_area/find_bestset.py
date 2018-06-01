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

features = ["_10min_not_fetched_order_num", "delivery_distance", "food_total_value",
            "10min#poi_id_mean_food_total_value", "_10min_working_rider_num", "poi_id_lasthour_mean_dd",
            "waiting_order_num", "dropoff_cluster#hour_mean_fetch_cost", "dropoff_cluster",
            "pickup_cluster#hour_mean_food_total_value", "dropoff_cluster#hour_mean_delivery_distance",
            "_avg_speed_hour#poi_id", "poi_id", "poi_id_waiting_order_num", "dropoff_cluster#10min_lasthour_min_dd",
            "_avg_speed_10min", "dropoff_cluster#hour_mean_dd", "pickup_cluster#10min_mean_box_total_value",
            "pickup_cluster#dropoff_cluster#hour_total_food_num",
            "pickup_cluster#dropoff_cluster#hour_waiting_order_num", "_avg_speed_weekday_hour",
            "10min#poi_id_lasthour_mean_dd", "pickup_cluster#dropoff_cluster#hour#10min_mean_dd",
            "pickup_cluster#10min_mean_food_total_value", "poi_id_lasthour_max_dd", "_10min_notbusy_working_rider_num",
            "_avg_speed_dropoff_cluster", "pickup_cluster#dropoff_cluster#hour#10min_mean_box_total_value",
            "hour#poi_id_coor_count", "10min#poi_id_lasthour_mean_fetch_cost", "pickup_cluster#hour_min_dd",
            "dropoff_cluster#10min_lasthour_mean_dd", "10min", "hour#poi_id_min_dd",
            "pickup_cluster#dropoff_cluster_lasthour_mean_dd", "10min#poi_id_lasthour_mean_arriveshop_cost",
            "pickup_cluster#dropoff_cluster#hour#10min_mean_arriveshop_cost",
            "dropoff_cluster#hour_mean_arriveshop_cost", "hour#poi_id_mean_arrive_guest_cost",
            "pickup_cluster#dropoff_cluster#hour_mean_fetch_cost", "pickup_cluster#10min_lasthour_mean_dd",
            "_avg_speed_pickup_cluster#dropoff_cluster#hour", "poi_lat", "poi_id_lasthour_mean_arriveshop_cost",
            "pickup_cluster#dropoff_cluster#hour_mean_box_total_value", "10min#poi_id_total_food_num",
            "pickup_cluster#dropoff_cluster_lasthour_mean_delivery_distance",
            "pickup_cluster#dropoff_cluster#hour_mean_food_total_value", "dropoff_cluster#hour_mean_arrive_guest_cost",
            "customer_longitude", "dropoff_cluster#hour_min_dd",
            "pickup_cluster#dropoff_cluster#hour#10min_mean_food_total_value",
            "pickup_cluster#dropoff_cluster#hour_mean_arrive_guest_cost",
            "pickup_cluster#dropoff_cluster#hour#10min_total_food_num", "pickup_cluster#hour_waiting_order_num",
            "pickup_cluster#10min_lasthour_max_dd", "pickup_cluster#dropoff_cluster#hour#10min_max_dd",
            "pickup_cluster#dropoff_cluster_lasthour_mean_arriveshop_cost", "pickup_cluster#10min_lasthour_min_dd",
            "dropoff_cluster#hour_waiting_order_num", "pickup_cluster", "hour#poi_id_mean_fetch_cost",
            "pickup_cluster#dropoff_cluster#hour_coor_count", "dropoff_cluster#hour_mean_box_total_value",
            "dropoff_cluster#hour_coor_count", "pickup_cluster#10min_waiting_order_num",
            "hour#poi_id_mean_food_total_value", "pickup_cluster#hour_mean_food_num",
            "pickup_cluster#dropoff_cluster_lasthour_mean_arrive_guest_cost",
            "10min#poi_id_lasthour_mean_delivery_distance", "direction", "poi_lng",
            "_avg_speed_10min#hour#dropoff_cluster", "hour#poi_id_mean_box_total_value",
            "pickup_cluster#dropoff_cluster#hour#10min_coor_count", "dropoff_cluster#10min_mean_food_num",
            "hour#poi_id_mean_dd", "pickup_cluster#dropoff_cluster#hour_mean_food_num",
            "pickup_cluster#10min_total_food_num", "pickup_cluster#10min_mean_food_num", "10min#poi_id_mean_food_num",
            "10min#poi_id_lasthour_min_dd", "pickup_cluster#hour_coor_count", "10min#poi_id_waiting_order_num",
            "pickup_cluster#hour_mean_arrive_guest_cost", "pickup_cluster#dropoff_cluster_lasthour_mean_fetch_cost",
            "hour#poi_id_max_dd", "pickup_cluster#dropoff_cluster_total_food_num", "_10min_deliverying_order_num",
            "hour#poi_id_mean_arriveshop_cost", "pickup_cluster#dropoff_cluster#hour#10min_mean_delivery_distance",
            "poi_id_mean_food_total_value", "hour#poi_id_waiting_order_num", "pickup_cluster#hour_mean_dd",
            "dropoff_cluster#10min_total_food_num", "pickup_cluster#dropoff_cluster_mean_food_total_value",
            "hour#poi_id_total_food_num", "pickup_cluster#dropoff_cluster#hour#10min_mean_food_num",
            "pickup_cluster#dropoff_cluster#hour_max_dd", "15min", "hour#poi_id_mean_food_num",
            "poi_id_lasthour_mean_arrive_guest_cost", "weekday", "dropoff_cluster#hour_total_food_num",
            "poi_id_lasthour_mean_delivery_distance", "10min#poi_id_lasthour_mean_arrive_guest_cost",
            "pickup_cluster#dropoff_cluster#hour_min_dd", "pickup_cluster#hour_total_food_num",
            "dropoff_cluster#hour_mean_food_num", "poi_id_total_food_num", "dropoff_cluster#10min_waiting_order_num",
            "poi_id_lasthour_min_dd", "_avg_speed_pickup_cluster", "pickup_cluster#hour_mean_arriveshop_cost",
            "dropoff_cluster#10min_mean_box_total_value", "customer_latitude", "_avg_speed_10min#hour#pickup_cluster",
            "pickup_cluster#10min_lasthour_mean_arrive_guest_cost",
            "pickup_cluster#dropoff_cluster#hour#10min_mean_fetch_cost",
            "dropoff_cluster#10min_lasthour_mean_fetch_cost", "pickup_cluster#dropoff_cluster_mean_food_num",
            "pickup_cluster#dropoff_cluster#hour_mean_arriveshop_cost", "pickup_cluster#hour_mean_delivery_distance",
            "pickup_cluster#dropoff_cluster_waiting_order_num",
            "pickup_cluster#dropoff_cluster#hour_mean_delivery_distance",
            "pickup_cluster#10min_lasthour_mean_delivery_distance",
            "pickup_cluster#dropoff_cluster#hour#10min_waiting_order_num",
            "pickup_cluster#dropoff_cluster#hour#10min_mean_arrive_guest_cost", "hour#poi_id_mean_delivery_distance",
            "_10min_wind", "area_id", "pickup_cluster#dropoff_cluster#hour_mean_dd",
            "pickup_cluster#10min_lasthour_mean_arriveshop_cost", "poi_id_mean_box_total_value",
            "pickup_cluster#hour_mean_box_total_value", "box_total_value", "10min#poi_id_mean_box_total_value",
            "pickup_cluster#hour_max_dd", "pickup_cluster#dropoff_cluster_mean_box_total_value", "food_num",
            "dropoff_cluster#10min_lasthour_mean_delivery_distance",
            "dropoff_cluster#10min_lasthour_mean_arrive_guest_cost", "poi_id_lasthour_mean_fetch_cost",
            "pickup_cluster#dropoff_cluster_lasthour_max_dd", "_10min_rain", "dropoff_cluster#10min_lasthour_max_dd",
            "poi_id_mean_food_num", "dropoff_cluster#10min_mean_food_total_value", "weekday_hour",
            "dropoff_cluster#hour_mean_food_total_value", "_avg_speed_weekday",
            "pickup_cluster#dropoff_cluster#hour#10min_min_dd", "10min#poi_id_lasthour_max_dd",
            "dropoff_cluster#hour_max_dd", "pickup_cluster#hour_mean_fetch_cost",
            "dropoff_cluster#10min_lasthour_mean_arriveshop_cost", "minute",
            "pickup_cluster#dropoff_cluster_lasthour_min_dd", "_10min_temperature",
            "pickup_cluster#10min_lasthour_mean_fetch_cost", "hour"]


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

    rs = []
    score = 0
    best_drop = []
    for f in [None] + features:
        print "try without f {}".format(f)
        se_features = list(set(features) - set([f]) - set(best_drop))
        lgb_train = lgb.Dataset(train_data[se_features], labels)
        lgb_eval = lgb.Dataset(test_data[se_features], labels_val, reference=lgb_train)

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
        gbm = lgb.train(params,
                        lgb_train,
                        valid_sets=lgb_eval,
                        num_boost_round=args.round,
                        early_stopping_rounds=500,
                        verbose_eval=False
                        )

        # print('Feature names:', gbm.feature_name())

        # print('Calculate feature importances...')
        # feature importances
        # print('Feature importances:', list(gbm.feature_importance()))

        df = pd.DataFrame({'feature': gbm.feature_name(), 'importances': gbm.feature_importance()})
        # print(df.sort_values('importances'))

        test_data.loc[:, "delivery_duration_prd"] = gbm.predict(test_data[se_features])
        cur_s = mean_absolute_error(test_data["delivery_duration"], test_data["delivery_duration_prd"])
        print "cur f is {},last score is {}, cur_s is {}, ".format(f, score, cur_s)
        if score == 0 or score > cur_s:
            score = cur_s
            if f:
                print "best drop f {}".format(f)
                best_drop.append(f)

        rs.append([f if f else "all_features",
                   mean_absolute_error(test_data["delivery_duration"], test_data["delivery_duration_prd"])])
    print "final score is  {},best_drop is {}".format(score, best_drop)
    # rs = order_id.merge(test_data[["order_id", "delivery_duration"]], left_on="order_id", right_on="order_id",
    #                     how="left")
    #
    # rs.to_csv(args.out_path, header=['order_id', 'delivery_duration'], index=False)
    # print sorted(rs, key=lambda x: -x[1])
    # print "\n".join(map(lambda x: ":".join(map(str, x)), sorted(rs, key=lambda x: -x[1])))
