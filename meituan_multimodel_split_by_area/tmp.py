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
import arrow as ar

# print(check_output(["ls", "./data"]).decode("utf8"))


# Any results you write to the current directory are saved as output.


from subprocess import check_output
import datetime
import arrow as ar
import pandas as pd
import numpy as np


# print(check_output(["ls", "./data"]).decode("utf8"))


# Any results you write to the current directory are saved as output.
# input_path = "/Users/dongjian/data/meituanKaggleData/"
# output_path = "/Users/dongjian/data/meituanKaggleData/"

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
    df.loc[mask, c] = df.loc[mask, c].apply(lambda x: ar.get(x).to("local"))
    df.loc[mask, 'day'] = df.loc[mask, c].apply(lambda x: x.day)
    df.loc[mask, 'weekday'] = df.loc[mask, c].apply(lambda x: x.weekday())
    df.loc[mask, 'hour'] = df.loc[mask, c].apply(lambda x: x.hour)
    df.loc[mask, '10min'] = df.loc[mask, c].apply(lambda x: x.minute / 10)
    df.loc[mask, 'high'] = df.loc[mask, c].apply(lambda x: 1 if x.hour in (11, 12, 13, 17, 18, 19) else 0)
    df.loc[mask, 'weekday_hour'] = df.loc[mask, c].apply(lambda x: x.weekday() * 24 + x.hour)
    df.loc[mask, c] = df.loc[mask, "tmp_order_unix_time"]
    return df


def load_area_data(file_name):
    df = pd.read_csv(file_name, dtype={'date': str, 'time': str})
    c = "log_unix_time"
    mask = pd.notnull(df[c])
    df.loc[mask, c] = df.loc[mask, c].apply(lambda x: ar.get(x).to("local"))
    df.loc[mask, 'day'] = df.loc[mask, c].apply(lambda x: x.day)
    df.loc[mask, 'weekday'] = df.loc[mask, c].apply(lambda x: x.weekday())
    df.loc[mask, 'hour'] = df.loc[mask, c].apply(lambda x: x.hour)
    df.loc[mask, '10min'] = df.loc[mask, c].apply(lambda x: x.minute / 10)
    df.loc[mask, 'minute'] = df.loc[mask, c].apply(lambda x: x.minute)
    df.loc[mask, 'high'] = df.loc[mask, c].apply(lambda x: 1 if x.hour in (11, 12, 13, 17, 18, 19) else 0)
    df.loc[mask, 'weekday_hour'] = df.loc[mask, c].apply(lambda x: x.weekday() * 24 + x.hour)
    df = df.loc[df.not_fetched_order_num > 0]
    return df


def load_weather_data(file_name):
    df = pd.read_csv(file_name, dtype={'date': str, 'time': str})
    c = "log_unix_time"
    mask = pd.notnull(df[c])
    df.loc[mask, c] = df.loc[mask, c].apply(lambda x: ar.get(x).to("local"))
    df.loc[mask, 'day'] = df.loc[mask, c].apply(lambda x: x.day)
    df.loc[mask, 'weekday'] = df.loc[mask, c].apply(lambda x: x.weekday())
    df.loc[mask, 'hour'] = df.loc[mask, c].apply(lambda x: x.hour)
    df.loc[mask, '10min'] = df.loc[mask, c].apply(lambda x: x.minute / 10)
    df.loc[mask, 'weekday_hour'] = df.loc[mask, c].apply(lambda x: x.weekday() * 24 + x.hour)
    df.loc[mask, 'high'] = df.loc[mask, c].apply(lambda x: 1 if x.hour in (11, 12, 13, 17, 18, 19) else 0)
    df = df.loc[df.temperature > -30]
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


def gene_mask(df, h, high=False):
    base = (df.delivery_duration < 4654.0) & (df.delivery_duration > 663.0) & (df.delivery_distance > 10)
    if high:
        return base & (df.high.values == 1)
    else:
        return base & ((df.hour.values == h) | (df.hour.values == h + 6))


def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


if __name__ == '__main__':
    args = init_arguments()
    waybill_info, area_realtime, weather_realtime, waybill_info_test_a, waybill_info_test_b, area_realtime_test, weather_realtime_test = load_data(
            args.input_path)
    wea_base_dict = {
        "temperature": {"_10min_temperature": "mean"},
        "wind": {"_10min_wind": "mean"},
        "rain": {"_10min_rain": "mean"},
    }
    are_base_dict = {
        "working_rider_num": {"_10min_working_rider_num": "mean"},
        "notbusy_working_rider_num": {"_10min_notbusy_working_rider_num": "mean"},
        "not_fetched_order_num": {"_10min_not_fetched_order_num": "mean"},
        "deliverying_order_num": {"_10min_deliverying_order_num": "mean"},
    }

    area_realtime_10min = nm.ka_add_groupby_features_1_vs_n(area_realtime, ["day","hour", "10min", "area_id"],
                                                            are_base_dict)
    weather_realtime_10min = nm.ka_add_groupby_features_1_vs_n(weather_realtime, ["day","hour", "10min", "area_id"],
                                                               wea_base_dict)

    ori = waybill_info.merge(area_realtime_10min, on=["day","hour", "10min", "area_id"], how='left') \
        .merge(weather_realtime_10min, on=["day","hour", "10min", "area_id"], how='left')

    print np.unique(ori.hour)

    pre = ori.loc[
        gene_mask(ori, 10) & (ori.day != 30) & (ori.day != 29) & (ori.day != 28) & (ori.day != 27) & (ori.day != 26) & (
            ori.day != 25) & (ori.day != 24)]

    tr = ori.loc[
        gene_mask(ori, 11) & (ori.day != 30) & (ori.day != 29) & (
            ori.day != 28) & (ori.day != 27) & (ori.day != 26) & (
            ori.day != 25) & (ori.day != 24)]

    ta = ori.loc[
        gene_mask(ori, 10) & (
            (ori.day == 30) | (ori.day == 29) | (ori.day == 28) | (ori.day == 27) | (ori.day == 26) | (
                ori.day == 25) | (
                ori.day == 24))]

    te = ori.loc[
        gene_mask(ori, 11) & (
            (ori.day == 30) | (ori.day == 29) | (ori.day == 28) | (ori.day == 27) | (ori.day == 26) | (
                ori.day == 25) | (
                ori.day == 24))]
    # print "pre.size,tr.size,te.size,ta.size ", len(pre), len(tr), len(te), len(ta)
    # # basic feature
    # print "start basic fea..."
    # tr["arriveshop_cost"] = tr["arriveshop_unix_time"].apply(lambda x: int(x)) - tr["tmp_order_unix_time"].apply(
    #         lambda x: int(x))
    # tr["fetch_cost"] = tr["fetch_unix_time"] - tr["arriveshop_unix_time"]
    # tr["arrive_guest_cost"] = tr["finish_unix_time"] - tr["fetch_unix_time"]
    #
    # tr.loc[:, "avg_speed"] = tr["delivery_distance"] / tr["delivery_duration"]
    # tr.loc[:, 'direction'] = bearing_array(tr['poi_lat'].values, tr['poi_lng'].values, tr['customer_latitude'].values,
    #                                        tr['customer_longitude'].values)
    # tr.loc[:, 'poi_lat_bin'] = np.round(tr['poi_lat'], 2)
    # tr.loc[:, 'poi_lng_bin'] = np.round(tr['poi_lng'], 2)
    # tr.loc[:, 'cst_lat_bin'] = np.round(tr['customer_latitude'], 2)
    # tr.loc[:, 'cst_lng_bin'] = np.round(tr['customer_longitude'], 2)
    #
    # te.loc[:, "avg_speed"] = tr["delivery_distance"] / tr["delivery_duration"]
    # te.loc[:, 'direction'] = bearing_array(te['poi_lat'].values, te['poi_lng'].values, te['customer_latitude'].values,
    #                                        te['customer_longitude'].values)
    # te.loc[:, 'poi_lat_bin'] = np.round(te['poi_lat'], 2)
    # te.loc[:, 'poi_lng_bin'] = np.round(te['poi_lng'], 2)
    # te.loc[:, 'cst_lat_bin'] = np.round(te['customer_latitude'], 2)
    # te.loc[:, 'cst_lng_bin'] = np.round(te['customer_longitude'], 2)
    #
    # tr["arriveshop_cost"] = tr["arriveshop_unix_time"].apply(lambda x: int(x)) - tr["tmp_order_unix_time"].apply(
    #         lambda x: int(x))
    # tr["fetch_cost"] = tr["fetch_unix_time"] - tr["arriveshop_unix_time"]
    # tr["arrive_guest_cost"] = tr["finish_unix_time"] - tr["fetch_unix_time"]
    # tr.loc[:, "avg_speed"] = tr["delivery_distance"] / tr["delivery_duration"]
    # print "done basic fea..."
    # # cluster
    #
    # print "start cluster fea..."
    # from sklearn.cluster import MiniBatchKMeans
    #
    # coords = np.vstack((tr[['poi_lat', 'poi_lng']].values,
    #                     tr[['customer_latitude', 'customer_longitude']].values,
    #                     te[['poi_lat', 'poi_lng']].values,
    #                     te[['customer_latitude', 'customer_longitude']].values))
    # sample_ind = np.random.permutation(len(coords))[:500000]
    # kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])
    # tr.loc[:, 'pickup_cluster'] = kmeans.predict(tr[['poi_lat', 'poi_lng']])
    # tr.loc[:, 'dropoff_cluster'] = kmeans.predict(tr[['customer_latitude', 'customer_longitude']])
    # te.loc[:, 'pickup_cluster'] = kmeans.predict(te[['poi_lat', 'poi_lng']])
    # te.loc[:, 'dropoff_cluster'] = kmeans.predict(te[['customer_latitude', 'customer_longitude']])
    # print "done cluster fea..."
    # for gby_col in ['hour', 'weekday', '10min',
    #                 'weekday_hour', 'pickup_cluster', 'dropoff_cluster']:
    #     speed_dict = {'avg_speed': {"_avg_speed_" + gby_col: "mean"
    #                                 }}
    #     tmp_d = nm.ka_add_groupby_features_1_vs_n(tr, [gby_col], speed_dict)
    #     tr = pd.merge(tr, tmp_d, how='left', on=gby_col, right_index=True)
    #     te = pd.merge(te, tmp_d, how='left', on=gby_col, right_index=True)
    #
    # for gby_cols in [['hour', 'pickup_cluster'], ['hour', 'dropoff_cluster'],
    #                  ['pickup_cluster', 'dropoff_cluster']]:
    #     speed_dict = {'avg_speed': {"_avg_speed_" + "_".join(gby_cols): "mean"},
    #                   "order_id": {"coor_count": "count"}}
    #     tmp_d = nm.ka_add_groupby_features_1_vs_n(tr, gby_cols, speed_dict)
    #     coord_stats = tmp_d[tmp_d['coor_count'] > 50]
    #     tr = pd.merge(tr, coord_stats, how='left', on=gby_cols)
    #     te = pd.merge(te, coord_stats, how='left', on=gby_cols)

    # poi_agg_dict = {'delivery_duration': {'_poi_max_dd': 'max',
    #                                       '_poi_min_dd': 'min',
    #                                       '_poi_mean_dd': 'median'},
    #                 "delivery_distance": {"_poi_mean_delivery_distance": "median"}
    #                 }
    # area_agg_dict = {'working_rider_num': {'_area_median_working_rider_num': 'median'},
    #                  "notbusy_working_rider_num": {"_area_median_notbusy_working_rider_num": "median"},
    #                  "not_fetched_order_num": {"_area_median_not_fetched_order_num": "median"},
    #                  "deliverying_order_num": {"_area_median_deliverying_order_num": "median"},
    #                  }

    # pre_poi_info = nm.ka_add_groupby_features_1_vs_n(pre, ['poi_id'], poi_agg_dict)
    # pre_area_info = nm.ka_add_groupby_features_1_vs_n(pre, ['area_id'], area_agg_dict)
    #
    # te_poi_info = nm.ka_add_groupby_features_1_vs_n(ta, ['poi_id'], poi_agg_dict)
    # te_area_info = nm.ka_add_groupby_features_1_vs_n(ta, ['area_id'], area_agg_dict)

    # print pre_poi_info.size, pre_area_info.size
    # tr = tr.merge(pre_poi_info, on="poi_id", how="left").merge(pre_area_info, on='area_id', how='left')
    print "tr.size()", len(tr)

    # te = te.merge(now_poi_info, on="poi_id", how="left").merge(now_area_info, on='area_id', how='left')
    print "te.size()", len(te)

    tr.to_csv(args.input_path + "train.csv")
    te.to_csv(args.input_path + "test.csv")
