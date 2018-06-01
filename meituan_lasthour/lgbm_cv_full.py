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
    "dropoff_cluster"
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
    features = list(set(train_data.columns.tolist()[1:]) - set(to_drop) - set(fea_drop))
    print "features,length {}, {}".format(len(features), features)


    # train_data.fillna(0, inplace=True)
    # test_data.fillna(0, inplace=True)

    order_id = pd.DataFrame(np.unique(test_data.order_id), columns=["order_id"])

    labels = train_data[args.label].values.astype(np.float32).flatten()
    labels_val = test_data[[args.label]].values.astype(np.float32).flatten()

    lgb_train = lgb.Dataset(train_data[features], labels)
    lgb_eval = lgb.Dataset(test_data[features], labels_val, reference=lgb_train)



    print('Start training...')


    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': "huber",
        'metric': {"mae"},
        'num_leaves': 256,
        'min_sum_hessian_in_leaf': 10,
        'max_depth': -12,
        'learning_rate': 0.03,
        'feature_fraction': 0.6,
        'verbose': 1,
    }

    def huber_approx_obj(preds, dtrain):
        d =preds - dtrain.get_label()    #remove .get_labels() for sklearn
        h = 2000  #h is delta in the graphic
        scale = 1 + (d / h) ** 2
        scale_sqrt = np.sqrt(scale)
        grad = d / scale_sqrt
        hess = 1 / scale / scale_sqrt
        return grad, hess

    def fair_obj(self, preds, dtrain):
        x = preds - dtrain.get_label()
        c = self.fair

        den = np.abs(x) * np.exp(self.fair_decay * self.iter) + c

        grad = c*x / den
        hess = c*c / den ** 2

        self.iter += 1

        return grad, hess

    gbm = lgb.train(params,
                    lgb_train,
                    fobj=huber_approx_obj,
                    valid_sets=lgb_eval,
                    num_boost_round=args.round,
                    verbose_eval=True)


    print('Feature names:', gbm.feature_name())

    print('Calculate feature importances...')
    # feature importances
    print('Feature importances:', list(gbm.feature_importance()))

    df = pd.DataFrame({'feature': gbm.feature_name(), 'importances': gbm.feature_importance()})
    print(df.sort_values('importances'))

    test_data.loc[:, "delivery_duration_prd"] = gbm.predict(test_data[features])

    print "!!!!!!!!!!!!!! rs is {} !!!!!!!!!!!!!!!!!".format(
            mean_absolute_error(test_data[args.label], test_data["delivery_duration_prd"]))
    print test_data["delivery_duration_prd"].describe()
    print test_data[args.label].describe()
    print "prd"
    print test_data["delivery_duration_prd"].describe()
    print "actual"
    print test_data[args.label].describe()

    test_data['mae'] = test_data[args.label] - test_data["delivery_duration_prd"]

    test_data.to_csv("/Users/dongjian/data/meituanKaggleData/test_out.csv")
    # rs = order_id.merge(test_data[["order_id", "delivery_duration"]], left_on="order_id", right_on="order_id",
    #                     how="left")
    #
    # rs.to_csv(args.out_path, header=['order_id', 'delivery_duration'], index=False)
