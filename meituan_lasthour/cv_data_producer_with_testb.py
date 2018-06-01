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
from sklearn.model_selection import train_test_split
# print(check_output(["ls", "./data"]).decode("utf8"))


# Any results you write to the current directory are saved as output.


from subprocess import check_output
import datetime
import arrow as ar
import pandas as pd
import numpy as np

# print(check_output(["ls", "./data"]).decode("utf8"))

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

    area_realtime_10min = nm.ka_add_groupby_features_1_vs_n(area_realtime, ["day", "hour", "minute", "area_id"],
                                                            are_base_dict)
    weather_realtime_10min = nm.ka_add_groupby_features_1_vs_n(weather_realtime, ["day", "hour", "15min", "area_id"],
                                                               wea_base_dict)
    ori = waybill_info.merge(area_realtime_10min, on=["day", "hour", "minute", "area_id"], how='left') \
        .merge(weather_realtime_10min, on=["day", "hour", "15min", "area_id"], how='left')
    ori.fillna("ffill")

    ori.loc[:, "area_busy_coef"] = ori["_10min_not_fetched_order_num"] / ori["_10min_notbusy_working_rider_num"]
    ori.loc[:, "bill_number_per_rider"] = ori["_10min_deliverying_order_num"] / ori["_10min_working_rider_num"]

    ori.loc[:, "avg_speed"] = ori["delivery_distance"] / ori["delivery_duration"]
    ori.loc[:, 'direction'] = bearing_array(ori['poi_lat'].values, ori['poi_lng'].values,
                                            ori['customer_latitude'].values,
                                            ori['customer_longitude'].values)
    ori.loc[:, 'poi_lat_bin'] = np.round(ori['poi_lat'], 3)
    ori.loc[:, 'poi_lng_bin'] = np.round(ori['poi_lng'], 3)
    ori.loc[:, 'cst_lat_bin'] = np.round(ori['customer_latitude'], 3)
    ori.loc[:, 'cst_lng_bin'] = np.round(ori['customer_longitude'], 3)
    # ori = ori.fillna(0)

    tr_ori, te_ori = train_test_split(ori, test_size=0.5, random_state=2017)
    tr_ori = tr_ori.loc[tr_ori.day != 30]
    te_ori = te_ori.loc[(te_ori.day == 30) | (te_ori.day <= 22)]

    tr, pre = tr_ori[gene_mask(tr_ori, 11) | gene_mask(tr_ori, 12) | gene_mask(tr_ori, 13)], tr_ori[
        gene_mask(tr_ori, 10)]
    te, ta = te_ori[gene_mask(te_ori, 11)], te_ori[gene_mask(te_ori, 10)]

    print "pre.size,tr.size,te.size,ta.size ", len(pre), len(tr), len(te), len(ta)
    # basic feature
    tr["arriveshop_cost"] = tr["arriveshop_unix_time"].apply(lambda x: int(x)) - tr["tmp_order_unix_time"].apply(
            lambda x: int(x))
    tr["fetch_cost"] = tr["fetch_unix_time"] - tr["arriveshop_unix_time"]
    tr["arrive_guest_cost"] = tr["finish_unix_time"] - tr["fetch_unix_time"]
    tr.loc[:, "arrive_guest_avg_speed"] = tr["delivery_distance"] / tr["arrive_guest_cost"]

    pre["arriveshop_cost"] = pre["arriveshop_unix_time"].apply(lambda x: int(x)) - pre["tmp_order_unix_time"].apply(
            lambda x: int(x))
    pre["fetch_cost"] = pre["fetch_unix_time"] - pre["arriveshop_unix_time"]
    pre["arrive_guest_cost"] = pre["finish_unix_time"] - pre["fetch_unix_time"]
    pre.loc[:, "arrive_guest_avg_speed"] = pre["delivery_distance"] / pre["arrive_guest_cost"]

    ta["arriveshop_cost"] = ta["arriveshop_unix_time"].apply(lambda x: int(x)) - ta["tmp_order_unix_time"].apply(
            lambda x: int(x))
    ta["fetch_cost"] = ta["fetch_unix_time"] - ta["arriveshop_unix_time"]
    ta["arrive_guest_cost"] = ta["finish_unix_time"] - ta["fetch_unix_time"]
    ta.loc[:, "arrive_guest_avg_speed"] = tr["delivery_distance"] / ta["arrive_guest_cost"]


    # cluster

    # print "start cluster fea..."
    # from sklearn.cluster import MiniBatchKMeans
    #
    # coords = np.vstack((tr[['poi_lat', 'poi_lng']].values,
    #                     tr[['customer_latitude', 'customer_longitude']].values,
    #                     te[['poi_lat', 'poi_lng']].values,
    #                     te[['customer_latitude', 'customer_longitude']].values))
    # sample_ind = np.random.permutation(len(coords))[:500000]
    # kmeans = MiniBatchKMeans(n_clusters=500, batch_size=10000).fit(coords[sample_ind])
    # tr.loc[:, 'pickup_cluster'] = kmeans.predict(tr[['poi_lat', 'poi_lng']])
    # tr.loc[:, 'dropoff_cluster'] = kmeans.predict(tr[['customer_latitude', 'customer_longitude']])
    # pre.loc[:, 'pickup_cluster'] = kmeans.predict(pre[['poi_lat', 'poi_lng']])
    # pre.loc[:, 'dropoff_cluster'] = kmeans.predict(pre[['customer_latitude', 'customer_longitude']])
    # te.loc[:, 'pickup_cluster'] = kmeans.predict(te[['poi_lat', 'poi_lng']])
    # te.loc[:, 'dropoff_cluster'] = kmeans.predict(te[['customer_latitude', 'customer_longitude']])
    # ta.loc[:, 'pickup_cluster'] = kmeans.predict(ta[['poi_lat', 'poi_lng']])
    # ta.loc[:, 'dropoff_cluster'] = kmeans.predict(ta[['customer_latitude', 'customer_longitude']])
    #
    # print "done cluster fea..."
    def gene_poi_agg_dict(prefix):
        return {
            'arrive_guest_avg_speed': {prefix + prefix_sep.join(gby_cols) + "_arrive_guest_avg_speed_mean": "median",
                                       prefix + prefix_sep.join(gby_cols) + "_arrive_guest_avg_speed_std": "std"},
            'avg_speed': {prefix + "_avg_speed_mean" + prefix_sep.join(gby_cols): "median",
                          prefix + "_avg_speed_std" + prefix_sep.join(gby_cols): "std"},
            'delivery_duration': {prefix + prefix_sep.join(gby_cols) + '_mean_dd': 'median',
                                  prefix + prefix_sep.join(gby_cols) + '_std_dd': 'std'},
            "arriveshop_cost": {prefix + prefix_sep.join(gby_cols) + "_mean_arriveshop_cost": "median",
                                prefix + prefix_sep.join(gby_cols) + "_std_arriveshop_cost": "std"},
            "fetch_cost": {prefix + prefix_sep.join(gby_cols) + "_mean_fetch_cost": "median",
                           prefix + prefix_sep.join(gby_cols) + "_std_fetch_cost": "std",
                           },
            "arrive_guest_cost": {prefix + prefix_sep.join(gby_cols) + "_mean_arrive_guest_cost": "median",
                                  prefix + prefix_sep.join(gby_cols) + "_std_arrive_guest_cost": "std"},
            "delivery_distance": {
                # prefix + prefix_sep.join(gby_cols) + "_mean_delivery_distance": "median",
                prefix + prefix_sep.join(gby_cols) + "_std_delivery_distance": "std"},
            "food_total_value": {
                # prefix + prefix_sep.join(gby_cols) + "_mean_food_total_value": "median",
                prefix + prefix_sep.join(gby_cols) + "_std_food_total_value": "std"},
            "box_total_value": {prefix + prefix_sep.join(gby_cols) + "_mean_box_total_value": "median"},
            # "food_num": {prefix + prefix_sep.join(gby_cols) + "_mean_food_num": "median",
            #              prefix + prefix_sep.join(gby_cols) + "_total_food_num": "sum"},
            "waiting_order_num": {prefix + prefix_sep.join(gby_cols) + "_mean_waiting_order_num": "median"},
            "order_id": {prefix + prefix_sep.join(gby_cols) + "_coor_count": "count",
                         }
        }


    for gby_cols in [['poi_id']]:
        prefix = "poi_agg_"
        poi_agg_dict = gene_poi_agg_dict(prefix)

        tmp_d = nm.ka_add_groupby_features_1_vs_n(tr, gby_cols, poi_agg_dict)
        coord_stats = tmp_d[tmp_d[prefix + prefix_sep.join(gby_cols) + "_coor_count"] > 10]
        tr = pd.merge(tr, coord_stats, how='left', on=gby_cols)
        te = pd.merge(te, coord_stats, how='left', on=gby_cols)

    for gby_cols in [
        ['poi_id']
        # ["poi_lat_bin",
        #  "poi_lng_bin",
        #  "cst_lat_bin",
        #  "cst_lng_bin"]
    ]:
        prefix = "lasthour_poi_agg_"
        lasthour_poi_agg_dict = gene_poi_agg_dict(prefix)

        pre_poi_info = nm.ka_add_groupby_features_1_vs_n(pre, gby_cols, lasthour_poi_agg_dict)
        ta_poi_info = nm.ka_add_groupby_features_1_vs_n(ta, gby_cols, lasthour_poi_agg_dict)
        tr = pd.merge(tr, pre_poi_info, how='left', on=gby_cols)
        te = pd.merge(te, ta_poi_info, how='left', on=gby_cols)

    for gby_cols in [
        ['area_id', 'hour', 'minute']
    ]:
        prefix = "area_id_"
        area_agg_dict = {
            "arriveshop_cost": {prefix + prefix_sep.join(gby_cols) + "_arriveshop_cost_mean": "median",
                                prefix + prefix_sep.join(gby_cols) + "_arriveshop_cost_std": "std"},
            "food_total_value": {
                # prefix + prefix_sep.join(gby_cols) + "_food_total_value_mean": "median",
                # prefix + prefix_sep.join(gby_cols) + "_food_total_value_std": "std",
                prefix + prefix_sep.join(gby_cols) + "_food_total_value_sum": "sum"}
            # "order_id"
            # "fetch_cost": {prefix + prefix_sep.join(gby_cols) + "_fetch_cost_mean": "median",
            #                prefix + prefix_sep.join(gby_cols) + "_fetch_cost_std": "std"},
            # "arrive_guest_cost": {prefix + prefix_sep.join(gby_cols) + "_aarrive_guest_cost_mean": "median",
            #                prefix + prefix_sep.join(gby_cols) + "_arrive_guest_cost_std": "std"}
        }

        tmp_d = nm.ka_add_groupby_features_1_vs_n(tr, gby_cols, area_agg_dict)
        tr = pd.merge(tr, tmp_d, how='left', on=gby_cols)
        te = pd.merge(te, tmp_d, how='left', on=gby_cols)

    tr['prd_arrive_guest_time'] = tr['delivery_distance'] / tr['poi_agg_poi_id_arrive_guest_avg_speed_mean']
    te['prd_arrive_guest_time'] = te['delivery_distance'] / te['poi_agg_poi_id_arrive_guest_avg_speed_mean']

    # tr11 = tr[gene_mask(tr, 11)]
    # te11 = te[gene_mask(te, 11)]

    # prefix = "cur_hour"
    # cur_hour_dict = {"delivery_distance": {prefix + prefix_sep.join(['poi_id']) + "_sum_delivery_distance": "sum",
    #                                        prefix + prefix_sep.join(['poi_id']) + "_std_delivery_distance": "std"},
    #                  "food_total_value": {prefix + prefix_sep.join(['poi_id']) + "_sum_food_total_value": "sum",
    #                                       prefix + prefix_sep.join(['poi_id']) + "_mean_food_total_value": "std"},
    #                  # "box_total_value": {prefix + prefix_sep.join(['poi_id']) + "_mean_box_total_value": "median"},
    #                  # "food_num": {prefix + prefix_sep.join(gby_cols) + "_mean_food_num": "median",
    #                  #              prefix + prefix_sep.join(gby_cols) + "_total_food_num": "sum"},
    #                  "waiting_order_num": {prefix + prefix_sep.join(['poi_id']) + "_sum_waiting_order_num": "sum"},
    #                  }
    # cur11 = nm.ka_add_groupby_features_1_vs_n(tr, ['poi_id', 'hour', '10min'], cur_hour_dict)
    # te_cur11 = nm.ka_add_groupby_features_1_vs_n(te, ['poi_id', 'hour', '10min'], cur_hour_dict)
    #
    # tr = pd.merge(tr, cur11, how='left', on=['poi_id', 'hour', '10min'])
    # te = pd.merge(te, te_cur11, how='left', on=['poi_id', 'hour', '10min'])

    # cur_area_dict = {"_10min_working_rider_num": {"cur_10min_working_rider_last": lambda x: x.iloc[-1],
    #                                               "cur_10min_working_rider_cha": lambda x: x.iloc[-1] - x.iloc[0]},
    #                  "_10min_notbusy_working_rider_num": {"cur_10min_notbusy_working_rider_last": lambda x: x.iloc[-1],
    #                                                       "cur_10min_notbusy_working_rider_cha": lambda x: x.iloc[-1] -
    #                                                                                                        x.iloc[0]},
    #                  "_10min_not_fetched_order_num": {"cur_10min_not_fetched_order_last": lambda x: x.iloc[-1],
    #                                                   "cur_10min_not_fetched_order_cha": lambda x: x.iloc[-1] - x.iloc[
    #                                                       0]
    #                                                   },
    #                  "_10min_deliverying_order_num": {
    #                      "cur_10min_deliverying_order_num_last": lambda x: x.iloc[-1],
    #                      "cur_10min_deliverying_order_num_cha": lambda x: x.iloc[-1] - x.iloc[0]
    #                  }
    #                  }
    #
    # area11 = nm.ka_add_groupby_features_1_vs_n(tr, ["area_id", 'hour', "10min"], cur_area_dict)
    # te_area_11 = nm.ka_add_groupby_features_1_vs_n(te, ["area_id", 'hour', "10min"], cur_area_dict)
    #
    # tr = pd.merge(tr, area11, how='left', on=["area_id", 'hour', "10min"])
    # te = pd.merge(te, te_area_11, how='left', on=["area_id", 'hour', "10min"])

    tr.to_csv(args.input_path + "train_cv.csv")
    te.to_csv(args.input_path + "test_cv.csv")
