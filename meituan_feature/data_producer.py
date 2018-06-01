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
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
import datetime


# print(check_output(["ls", "./data"]).decode("utf8"))


# Any results you write to the current directory are saved as output.

def init_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, dest='input_path',
                        help="data")
    parser.add_argument('--output', type=str, dest='output_path',
                        help="data")
    parser.add_argument('-k', type=str, dest="key",
                        help="map key(rider or rider_poi)")
    return parser.parse_args()


def load_order_data(file_name, sep=","):
    df = pd.read_csv('%s' % file_name, sep)
    names = map(lambda x: x.replace("algocamp_order_test.", ""), df.columns.tolist())

    df.columns = names
    # df.rename(columns=names,inplace=True)
    c = 'order_unix_time'
    mask = pd.notnull(df[c])
    df.loc[mask, "tmp_order_unix_time"] = df.loc[mask, c]
    df.loc[mask, c] = df.loc[mask, c].apply(lambda x: datetime.datetime.fromtimestamp(x))
    df.loc[mask, 'date'] = df.loc[mask, c].apply(lambda x: x.strftime('%Y%m%d'))
    df.loc[mask, 'hour'] = df.loc[mask, c].apply(lambda x: x.hour)
    df.loc[mask, 'minute'] = df.loc[mask, c].apply(lambda x: x.minute)
    df.loc[mask, c] = df.loc[mask, "tmp_order_unix_time"]
    # print file_name
    # print df.head(5)
    # print df['tmp_order_unix_time'].head(5)

    return df


def load_area_data(file_name):
    df = pd.read_csv(file_name, dtype={'date': str, 'time': str})
    mask = pd.notnull(df['time'])
    df.loc[mask, 'hour'] = df.loc[mask, 'time'].apply(lambda x: int(x[:2]))
    df.loc[mask, 'minute'] = df.loc[mask, 'time'].apply(lambda x: int(x[2:]))
    df.drop(['log_unix_time', 'time'], axis=1, inplace=True)
    # df['not_fetched_order_num'] = df['not_fetched_order_num'].apply(lambda x: x if int(x) > 0 else 0)
    return df


def load_weather_data(file_name):
    df = pd.read_csv(file_name, dtype={'date': str, 'time': str})
    mask = pd.notnull(df['time'])
    df.loc[mask, 'hour'] = df.loc[mask, 'time'].apply(lambda x: int(x[:2]))
    df.loc[mask, 'minute'] = df.loc[mask, 'time'].apply(lambda x: int(x[2:]))
    df.drop(['log_unix_time', 'time'], axis=1, inplace=True)
    return df
def load_data(path_data):
    '''
    --------------------------------order_product--------------------------------
    * Unique in order_id + product_id
    '''
    waybill_info = load_order_data(path_data + 'waybill_info.csv')
    area_realtime = load_area_data(path_data + 'area_realtime.csv')
    weather_realtime = load_weather_data(path_data + 'weather_realtime.csv')
    waybill_info_test_a = load_order_data(path_data + 'waybill_info_test_a.csv', sep="\t")
    waybill_info_test_b = load_order_data(path_data + 'waybill_info_test_b.csv')
    area_realtime_test = load_area_data(path_data + 'area_realtime_test.csv')
    weather_realtime_test = load_weather_data(path_data + 'weather_realtime_test.csv')

    return waybill_info, area_realtime, weather_realtime, waybill_info_test_a, waybill_info_test_b, area_realtime_test, weather_realtime_test


if __name__ == '__main__':
    args = init_arguments()
    waybill_info, area_realtime, weather_realtime, waybill_info_test_a, waybill_info_test_b, area_realtime_test, weather_realtime_test = load_data(
            args.input_path)

    tr = waybill_info.merge(weather_realtime, on=['date', 'hour', 'minute', 'area_id'], how='left') \
        .merge(area_realtime, on=['date', 'hour', 'minute', 'area_id'], how='left')
    mask = (tr.delivery_duration < 4654.0) & (tr.delivery_duration > 663.0) \
           & (
               #  (tr.hour.values == 9)
               #  (tr.hour.values == 10)
               (tr.hour.values == 11)
               | (tr.hour.values == 12)
               | (tr.hour.values == 13)
               # | (tr.hour.values == 14)
               # | (tr.hour.values == 15)
               # | (tr.hour.values == 16)
               | (tr.hour.values == 17)
               | (tr.hour.values == 18)
               | (tr.hour.values == 19)
               # | (tr.hour.values == 20)
           )
    tr = tr.loc[mask]
    print "tr head 5"
    print tr.head(5)
    te = waybill_info_test_b.merge(weather_realtime_test, on=['date', 'hour', 'minute', 'area_id'], how='left') \
        .merge(area_realtime_test, on=['date', 'hour', 'minute', 'area_id'], how='left')

    # tr["arriveshop_cost"] = tr["arriveshop_unix_time"].apply(lambda x: int(x)) - tr["tmp_order_unix_time"].apply(
    #     lambda x: int(x))
    # tr["fetch_cost"] = tr["fetch_unix_time"] - tr["arriveshop_unix_time"]
    # tr["arrive_guest_cost"] = tr["finish_unix_time"] - tr["fetch_unix_time"]
    #
    # poi_agg_dict = {'delivery_duration': {'_poi_max_dd': 'max',
    #                                       '_poi_min_dd': 'min',
    #                                       '_poi_mean_dd': 'median'},
    #                 "arriveshop_cost": {"_poi_mean_arriveshop_cost": "median"},
    #                 "fetch_cost": {"_poi_mean_fetch_cost": "median",},
    #                 "arrive_guest_cost": {"_poi_mean_arrive_guest_cost": "median"},
    #                 "delivery_distance": {"_poi_mean_delivery_distance": "median"}
    #                 }
    #
    # poi_info = nm.ka_add_groupby_features_1_vs_n(tr, ['poi_id'], poi_agg_dict)
    # area_agg_dict = {'working_rider_num': {'_area_median_working_rider_num': 'median'},
    #                  "notbusy_working_rider_num": {"_area_median_notbusy_working_rider_num": "median"},
    #                  "not_fetched_order_num": {"_area_median_not_fetched_order_num": "median",},
    #                  "deliverying_order_num": {"_area_median_deliverying_order_num": "median"},
    #                  }
    # area_info = nm.ka_add_groupby_features_1_vs_n(tr, ['area_id'], area_agg_dict)
    #
    # tr = tr.merge(poi_info, on='poi_id', how="left").merge(area_info, on='area_id', how="left")
    # te = te.merge(poi_info, on='poi_id', how="left").merge(area_info, on='area_id', how="left")
    #
    # tr["above_avg_distance"] = tr["_poi_mean_delivery_distance"] - tr["delivery_distance"]
    # te["above_avg_distance"] = te["_poi_mean_delivery_distance"] - te["delivery_distance"]
    # tr["_area_not_fetched_order_num_above_avg"] = tr["not_fetched_order_num"] - tr["_area_median_not_fetched_order_num"]
    # te["_area_not_fetched_order_num_above_avg"] = te["not_fetched_order_num"] - te["_area_median_not_fetched_order_num"]

    tr.to_csv(args.input_path + "train.csv")
    te.to_csv(args.input_path + "test.csv")
