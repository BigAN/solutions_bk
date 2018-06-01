# encoding:utf8
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
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

prefix_sep = "#"


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
    df.loc[mask, '15min'] = df.loc[mask, c].apply(lambda x: x.minute / 15)
    df.loc[mask, 'minute'] = df.loc[mask, c].apply(lambda x: x.minute)
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
    df.loc[mask, '15min'] = df.loc[mask, c].apply(lambda x: x.minute / 15)
    df.loc[mask, 'minute'] = df.loc[mask, c].apply(lambda x: x.minute)
    df.loc[mask, 'high'] = df.loc[mask, c].apply(lambda x: 1 if x.hour in (11, 12, 13, 17, 18, 19) else 0)
    df.loc[mask, 'weekday_hour'] = df.loc[mask, c].apply(lambda x: x.weekday() * 24 + x.hour)
    df = df.loc[df.not_fetched_order_num > 0]
    df['busy_working_rider_num'] = df['working_rider_num'] - df['notbusy_working_rider_num']

    return df


def load_weather_data(file_name):
    df = pd.read_csv(file_name, dtype={'date': str, 'time': str})
    c = "log_unix_time"
    mask = pd.notnull(df[c])
    df.loc[mask, c] = df.loc[mask, c].apply(lambda x: ar.get(x).to("local"))
    df.loc[mask, 'day'] = df.loc[mask, c].apply(lambda x: x.day)
    df.loc[mask, 'weekday'] = df.loc[mask, c].apply(lambda x: x.weekday())
    df.loc[mask, 'hour'] = df.loc[mask, c].apply(lambda x: x.hour)
    df.loc[mask, '15min'] = df.loc[mask, c].apply(lambda x: x.minute / 15)
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
    base = (df.delivery_duration < 4654.0) & (df.delivery_duration > 663.0) & \
           (df.delivery_distance > 10)  # ???
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
    waybill_info_test_a.columns = map(lambda x: x.replace("algocamp_order_test.", ""),
                                      waybill_info_test_a.columns.tolist())

    wea_base_dict = {
        "temperature": {"_10min_temperature": "mean"},
        "wind": {"_10min_wind": "mean"},
        "rain": {"_10min_rain": "mean"},
    }
    are_base_dict = {
        "working_rider_num": {"_10min_working_rider_num": "mean"},
        "notbusy_working_rider_num": {"_10min_notbusy_working_rider_num": "mean"},
        # "busy_working_rider_num":{"_10min_busy_working_rider_num":"mean"},
        "not_fetched_order_num": {"_10min_not_fetched_order_num": "mean"},
        "deliverying_order_num": {"_10min_deliverying_order_num": "mean"},
        # "all_order_area_num":{"_10min_all_order_area_num": "mean"}
    }

    area_realtime_10min = nm.ka_add_groupby_features_1_vs_n(area_realtime, ["day", "hour", "minute", "area_id"],
                                                            are_base_dict)
    area_realtime_10min['busy_working_rider_num'] = area_realtime_10min["_10min_working_rider_num"] - \
                                                    area_realtime_10min['_10min_notbusy_working_rider_num']

    weather_realtime_10min = nm.ka_add_groupby_features_1_vs_n(weather_realtime, ["day", "hour", "15min", "area_id"],
                                                               wea_base_dict)
    area_realtime_10min_test = nm.ka_add_groupby_features_1_vs_n(area_realtime_test,
                                                                 ["day", "hour", "minute", "area_id"],
                                                                 are_base_dict)
    area_realtime_10min_test['busy_working_rider_num'] = area_realtime_10min_test["_10min_working_rider_num"] - \
                                                         area_realtime_10min_test['_10min_notbusy_working_rider_num']

    weather_realtime_10min_test = nm.ka_add_groupby_features_1_vs_n(weather_realtime_test,
                                                                    ["day", "hour", "15min", "area_id"],
                                                                    wea_base_dict)

    waybill_info_test_a = waybill_info_test_a.merge(area_realtime_10min_test, on=["day", "hour", "minute", "area_id"],
                                                    how='left').merge(weather_realtime_10min_test,
                                                                      on=["day", "hour", "15min", "area_id"],
                                                                      how='left')

    tr = waybill_info.merge(area_realtime_10min, on=["day", "hour", "minute", "area_id"], how='left') \
        .merge(weather_realtime_10min, on=["day", "hour", "15min", "area_id"], how='left')

    te = waybill_info_test_b.merge(area_realtime_10min_test, on=["day", "hour", "minute", "area_id"], how='left') \
        .merge(weather_realtime_10min_test, on=["day", "hour", "15min", "area_id"], how='left')

    pre = tr.loc[gene_mask(tr, 10)]

    ta = waybill_info_test_a[gene_mask(waybill_info_test_a, 10)]
    tr = tr.loc[gene_mask(tr, 0, True)]

    # basic feature

    print "start basic fea..."
    tr.loc[:, 'direction'] = bearing_array(tr['poi_lat'].values, tr['poi_lng'].values,
                                           tr['customer_latitude'].values,
                                           tr['customer_longitude'].values)
    tr.loc[:, 'poi_lat_bin'] = np.round(tr['poi_lat'], 3)
    tr.loc[:, 'poi_lng_bin'] = np.round(tr['poi_lng'], 3)
    tr.loc[:, 'cst_lat_bin'] = np.round(tr['customer_latitude'], 3)
    tr.loc[:, 'cst_lng_bin'] = np.round(tr['customer_longitude'], 3)

    # te.loc[:, "avg_speed"] = te["delivery_distance"] / te["delivery_duration"]
    te.loc[:, 'direction'] = bearing_array(te['poi_lat'].values, te['poi_lng'].values,
                                           te['customer_latitude'].values,
                                           te['customer_longitude'].values)
    te.loc[:, 'poi_lat_bin'] = np.round(te['poi_lat'], 3)
    te.loc[:, 'poi_lng_bin'] = np.round(te['poi_lng'], 3)
    te.loc[:, 'cst_lat_bin'] = np.round(te['customer_latitude'], 3)
    te.loc[:, 'cst_lng_bin'] = np.round(te['customer_longitude'], 3)

    tr["arriveshop_cost"] = tr["arriveshop_unix_time"].apply(lambda x: int(x)) - tr["tmp_order_unix_time"].apply(
            lambda x: int(x))
    tr["fetch_cost"] = tr["fetch_unix_time"] - tr["arriveshop_unix_time"]
    tr["arrive_guest_cost"] = tr["finish_unix_time"] - tr["fetch_unix_time"]
    tr.loc[:, "avg_speed"] = tr["delivery_distance"] / tr["arrive_guest_cost"]
    tr = tr.loc[tr['avg_speed'] < 10]

    print "cal pre cost"
    pre["arriveshop_cost"] = pre["arriveshop_unix_time"].apply(lambda x: int(x)) - pre["tmp_order_unix_time"].apply(
            lambda x: int(x))
    pre["fetch_cost"] = pre["fetch_unix_time"] - pre["arriveshop_unix_time"]
    pre["arrive_guest_cost"] = pre["finish_unix_time"] - pre["fetch_unix_time"]
    pre.loc[:, "avg_speed"] = pre["delivery_distance"] / pre["arrive_guest_cost"]
    pre = pre.loc[pre['avg_speed'] < 10]

    print "cal ta cost"
    ta["arriveshop_cost"] = ta["arriveshop_unix_time"].apply(lambda x: int(x)) - ta["tmp_order_unix_time"].apply(
            lambda x: int(x))
    ta["fetch_cost"] = ta["fetch_unix_time"] - ta["arriveshop_unix_time"]
    ta["arrive_guest_cost"] = ta["finish_unix_time"] - ta["fetch_unix_time"]
    ta.loc[:, "avg_speed"] = ta["delivery_distance"] / ta["arrive_guest_cost"]
    ta = ta.loc[ta['avg_speed'] < 10]
    print "done basic fea..."
    # cluster





    print "start cluster fea..."
    from sklearn.cluster import MiniBatchKMeans

    coords = np.vstack((tr[['poi_lat', 'poi_lng']].values,
                        tr[['customer_latitude', 'customer_longitude']].values,
                        te[['poi_lat', 'poi_lng']].values,
                        te[['customer_latitude', 'customer_longitude']].values))
    sample_ind = np.random.permutation(len(coords))[:1200000]
    kmeans = MiniBatchKMeans(n_clusters=500, batch_size=10000).fit(coords[sample_ind])
    tr.loc[:, 'pickup_cluster'] = kmeans.predict(tr[['poi_lat', 'poi_lng']])
    tr.loc[:, 'dropoff_cluster'] = kmeans.predict(tr[['customer_latitude', 'customer_longitude']])
    pre.loc[:, 'pickup_cluster'] = kmeans.predict(pre[['poi_lat', 'poi_lng']])
    pre.loc[:, 'dropoff_cluster'] = kmeans.predict(pre[['customer_latitude', 'customer_longitude']])
    te.loc[:, 'pickup_cluster'] = kmeans.predict(te[['poi_lat', 'poi_lng']])
    te.loc[:, 'dropoff_cluster'] = kmeans.predict(te[['customer_latitude', 'customer_longitude']])
    ta.loc[:, 'pickup_cluster'] = kmeans.predict(ta[['poi_lat', 'poi_lng']])
    ta.loc[:, 'dropoff_cluster'] = kmeans.predict(ta[['customer_latitude', 'customer_longitude']])

    print "done cluster fea..."

    print "speed agg fea..."
    for gby_cols in [['weekday'], ['10min'],
                     ['weekday_hour'], ['pickup_cluster'], ['dropoff_cluster'],
                     ['10min', 'weekday_hour', 'pickup_cluster'],
                     ['10min', 'weekday_hour', 'dropoff_cluster'],
                     ['pickup_cluster', 'dropoff_cluster'],
                     ['poi_id'],
                     ['poi_id', 'weekday'],
                     ['poi_id', 'day']]:
        poi_agg_dict = {
            'avg_speed': {"_avg_speed_" + prefix_sep.join(gby_cols): "mean"},
            'delivery_duration': {
                prefix_sep.join(gby_cols) + '_mean_dd': 'median'},
            "arriveshop_cost": {prefix_sep.join(gby_cols) + "_mean_arriveshop_cost": "median"},
            "fetch_cost": {prefix_sep.join(gby_cols) + "_mean_fetch_cost": "median",},
            "arrive_guest_cost": {prefix_sep.join(gby_cols) + "_mean_arrive_guest_cost": "median"},
            "delivery_distance": {prefix_sep.join(gby_cols) + "_mean_delivery_distance": "median"},
            "food_total_value": {prefix_sep.join(gby_cols) + "_mean_food_total_value": "median"},
            "box_total_value": {prefix_sep.join(gby_cols) + "_mean_box_total_value": "median"},
            "food_num": {prefix_sep.join(gby_cols) + "_mean_food_num": "median",
                         prefix_sep.join(gby_cols) + "_total_food_num": "sum"},
            "waiting_order_num": {prefix_sep.join(gby_cols) + "_waiting_order_num": "median",
                                  prefix_sep.join(gby_cols) + "_waiting_order_num": "sum"},
            "_10min_working_rider_num": {
                prefix_sep.join(gby_cols) + "_10min_working_rider_num": "median",
            },
            "_10min_notbusy_working_rider_num": {
                prefix_sep.join(gby_cols) + "_10min_notbusy_working_rider_num": "median",
            },
            "_10min_not_fetched_order_num": {
                prefix_sep.join(gby_cols) + "_10min_not_fetched_order_num": "median",
            },
            "_10min_deliverying_order_num": {
                prefix_sep.join(gby_cols) + "_10min_not_fetched_order_num": "median",
            },
            "order_id": {prefix_sep.join(gby_cols) + "_coor_count": "count"}
        }
        tmp_d = nm.ka_add_groupby_features_1_vs_n(tr, gby_cols, poi_agg_dict)
        coord_stats = tmp_d[tmp_d[prefix_sep.join(gby_cols) + "_coor_count"] >= 20]
        tr = pd.merge(tr, coord_stats, how='left', on=gby_cols)
        te = pd.merge(te, coord_stats, how='left', on=gby_cols)

    # print "pre poi agg fea..."

    for gby_cols in [
        ['10min', 'hour', 'weekday', 'poi_id'], ['poi_id', 'hour', 'weekday'], ['pickup_cluster', '10min'],
        ['dropoff_cluster', '10min'],
        ['pickup_cluster', 'dropoff_cluster']
    ]:
        pre_poi_agg_dict = {'delivery_duration':
                                {prefix_sep.join(gby_cols) + "_lasthour" + '_mean_dd': 'median'},
                            "arriveshop_cost": {
                                prefix_sep.join(gby_cols) + "_lasthour" + "_mean_arriveshop_cost": "median"},
                            "fetch_cost": {prefix_sep.join(gby_cols) + "_lasthour" + "_mean_fetch_cost": "median",},
                            "arrive_guest_cost": {
                                prefix_sep.join(gby_cols) + "_lasthour" + "_mean_arrive_guest_cost": "median"},
                            "delivery_distance": {
                                prefix_sep.join(gby_cols) + "_lasthour" + "_mean_delivery_distance": "median"},
                            "food_total_value": {prefix_sep.join(gby_cols) + "_mean_food_total_value": "median"},
                            "box_total_value": {prefix_sep.join(gby_cols) + "_mean_box_total_value": "median"},
                            "food_num": {prefix_sep.join(gby_cols) + "_mean_food_num": "median",
                                         prefix_sep.join(gby_cols) + "_total_food_num": "sum"},
                            "waiting_order_num": {prefix_sep.join(gby_cols) + "_waiting_order_num": "median",
                                                  },
                            "_10min_working_rider_num": {
                                prefix_sep.join(gby_cols) + "_lasthour" + "_10min_working_rider_num": "median",
                            },
                            "_10min_notbusy_working_rider_num": {
                                prefix_sep.join(gby_cols) + "_lasthour" + "_10min_notbusy_working_rider_num": "median",
                            },
                            "_10min_not_fetched_order_num": {
                                prefix_sep.join(gby_cols) + "_lasthour" + "_10min_not_fetched_order_num": "median",
                            },
                            "_10min_deliverying_order_num": {
                                prefix_sep.join(gby_cols) + "_lasthour" + "_10min_not_fetched_order_num": "median",
                            },
                            # "order_id": {prefix_sep.join(gby_cols) + "_coor_count": "count"}
                            }
        pre_poi_info = nm.ka_add_groupby_features_1_vs_n(pre, gby_cols, pre_poi_agg_dict)
        ta_poi_info = nm.ka_add_groupby_features_1_vs_n(ta, gby_cols, pre_poi_agg_dict)
        # coord_stats = tmp_d[tmp_d[prefix_sep.join(gby_cols) + "_coor_count"] >= 10]
        tr = pd.merge(tr, pre_poi_info, how='left', on=gby_cols)
        te = pd.merge(te, ta_poi_info, how='left', on=gby_cols)

    # print "area info"
    for gby_cols in [
        ['10min', 'hour', 'weekday', 'area_id']
    ]:
        area_info_dict = {'avg_speed': {prefix_sep.join(gby_cols) + '_mean_dd': 'median'},
                          "arriveshop_cost": {
                              prefix_sep.join(gby_cols) + "_mean_arriveshop_cost": "median"},
                          'delivery_duration':
                              {prefix_sep.join(gby_cols) + "_lasthour" + '_mean_dd': 'median'},
                          "fetch_cost": {prefix_sep.join(gby_cols) + "_lasthour" + "_mean_fetch_cost": "median",},
                          "arrive_guest_cost": {
                              prefix_sep.join(gby_cols) + "_lasthour" + "_mean_arrive_guest_cost": "median"},
                          "order_id": {prefix_sep.join(gby_cols) + "_coor_count": "count"}
                          }
        tmp_d = nm.ka_add_groupby_features_1_vs_n(tr, gby_cols, area_info_dict)
        coord_stats = tmp_d[tmp_d[prefix_sep.join(gby_cols) + "_coor_count"] >= 20]
        tr = pd.merge(tr, coord_stats, how='left', on=gby_cols)
        te = pd.merge(te, coord_stats, how='left', on=gby_cols)

    print "pre_poi_info {} , ta_poi_info {}".format(len(pre_poi_info), len(ta_poi_info))
    # tr = tr.merge(pre_poi_info, on="poi_id", how="left").merge(pre_area_info, on='area_id', how='left')
    print "tr.size()", len(tr)

    # te = te.merge(now_poi_info, on="poi_id", how="left").merge(now_area_info, on='area_id', how='left')
    print "te.size()", len(te)

    tr.to_csv(args.input_path + "train_sub.csv")
    te.to_csv(args.input_path + "test_sub.csv")
