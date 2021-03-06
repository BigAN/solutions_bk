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


def parse_to_min(x):
    hour = x[:2]
    min = x[-2:]
    return int(hour) * 60 + int(min)


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
    df.loc[mask, 'cur_min'] = df.loc[mask, c].apply(lambda x: parse_to_min(x.format("HHmm")))
    df.loc[mask, 'next15_min'] = df.loc[mask, c].apply(lambda x: parse_to_min(x.replace(minute=+15).format("HHmm")))
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
    area_realtime_10min_test = nm.ka_add_groupby_features_1_vs_n(area_realtime_test,
                                                                 ["day", "hour", "minute", "area_id"],
                                                                 are_base_dict)
    weather_realtime_10min_test = nm.ka_add_groupby_features_1_vs_n(weather_realtime_test,
                                                                    ["day", "hour", "15min", "area_id"],
                                                                    wea_base_dict)

    tr = waybill_info.merge(area_realtime_10min, on=["day", "hour", "minute", "area_id"], how='left') \
        .merge(weather_realtime_10min, on=["day", "hour", "15min", "area_id"], how='left')

    te = waybill_info_test_b.merge(area_realtime_10min_test, on=["day", "hour", "minute", "area_id"], how='left') \
        .merge(weather_realtime_10min_test, on=["day", "hour", "15min", "area_id"], how='left')

    pre = tr.loc[gene_mask(tr, 10)]

    waybill_info_test_a.columns = map(lambda x: x.replace("algocamp_order_test.", ""),
                                      waybill_info_test_a.columns.tolist())
    ta = waybill_info_test_a[gene_mask(waybill_info_test_a, 10)]
    tr = tr.loc[gene_mask(tr, 0, True)]

    # basic feature

    print "start basic fea..."

    tr.loc[:, "area_busy_coef"] = tr["_10min_not_fetched_order_num"] / tr["_10min_notbusy_working_rider_num"]
    tr.loc[:, "bill_number_per_rider"] = tr["_10min_deliverying_order_num"] / tr["_10min_working_rider_num"]

    tr.loc[:, "avg_speed"] = tr["delivery_distance"] / tr["delivery_duration"]
    tr.loc[:, 'direction'] = bearing_array(tr['poi_lat'].values, tr['poi_lng'].values,
                                           tr['customer_latitude'].values,
                                           tr['customer_longitude'].values)
    tr.loc[:, 'poi_lat_bin'] = np.round(tr['poi_lat'], 3)
    tr.loc[:, 'poi_lng_bin'] = np.round(tr['poi_lng'], 3)
    tr.loc[:, 'cst_lat_bin'] = np.round(tr['customer_latitude'], 3)
    tr.loc[:, 'cst_lng_bin'] = np.round(tr['customer_longitude'], 3)
    # tr.fill

    # te.loc[:, "avg_speed"] = te["delivery_distance"] / te["delivery_duration"]
    te.loc[:, "area_busy_coef"] = te["_10min_not_fetched_order_num"] / te["_10min_notbusy_working_rider_num"]
    te.loc[:, "bill_number_per_rider"] = te["_10min_deliverying_order_num"] / te["_10min_working_rider_num"]
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
    tr.loc[:, "arrive_guest_avg_speed"] = tr["delivery_distance"] / tr["arrive_guest_cost"]

    print "cal pre cost"
    pre["arriveshop_cost"] = pre["arriveshop_unix_time"].apply(lambda x: int(x)) - pre["tmp_order_unix_time"].apply(
            lambda x: int(x))
    pre["fetch_cost"] = pre["fetch_unix_time"] - pre["arriveshop_unix_time"]
    pre["arrive_guest_cost"] = pre["finish_unix_time"] - pre["fetch_unix_time"]
    pre.loc[:, "avg_speed"] = pre["delivery_distance"] / pre["delivery_duration"]
    pre.loc[:, "arrive_guest_avg_speed"] = pre["delivery_distance"] / pre["arrive_guest_cost"]
    tr.loc[:, "area_busy_coef"] = tr["_10min_not_fetched_order_num"] / tr["_10min_notbusy_working_rider_num"]
    tr.loc[:, "bill_number_per_rider"] = tr["_10min_deliverying_order_num"] / tr["_10min_working_rider_num"]

    print "cal ta cost"
    ta["arriveshop_cost"] = ta["arriveshop_unix_time"].apply(lambda x: int(x)) - ta["tmp_order_unix_time"].apply(
            lambda x: int(x))
    ta["fetch_cost"] = ta["fetch_unix_time"] - ta["arriveshop_unix_time"]
    ta["arrive_guest_cost"] = ta["finish_unix_time"] - ta["fetch_unix_time"]
    ta.loc[:, "avg_speed"] = ta["delivery_distance"] / ta["delivery_duration"]
    ta.loc[:, "arrive_guest_avg_speed"] = tr["delivery_distance"] / ta["arrive_guest_cost"]

    print "done basic fea..."
    # cluster

    print "poi agg fea..."


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
                prefix + prefix_sep.join(gby_cols) + "_mean_delivery_distance": "median",
                prefix + prefix_sep.join(gby_cols) + "_std_delivery_distance": "std"
            },
            "food_total_value": {
                prefix + prefix_sep.join(gby_cols) + "_mean_food_total_value": "median",
                prefix + prefix_sep.join(gby_cols) + "_std_food_total_value": "std"
            },
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
    # from sklearn.cluster import MiniBatchKMeans

    # coords = np.vstack((tr[['poi_lat', 'poi_lng']].values,
    #                     tr[['customer_latitude', 'customer_longitude']].values,
    #                     te[['poi_lat', 'poi_lng']].values,
    #                     te[['customer_latitude', 'customer_longitude']].values))
    # sample_ind = np.random.permutation(len(coords))[:1200000]
    # kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])
    # tr.loc[:, 'pickup_cluster'] = kmeans.predict(tr[['poi_lat', 'poi_lng']])
    # tr.loc[:, 'dropoff_cluster'] = kmeans.predict(tr[['customer_latitude', 'customer_longitude']])
    # te.loc[:, 'pickup_cluster'] = kmeans.predict(te[['poi_lat', 'poi_lng']])
    # te.loc[:, 'dropoff_cluster'] = kmeans.predict(te[['customer_latitude', 'customer_longitude']])

    # for gby_cols in [
    #     # ['dropoff_cluster'],
    #     ['pickup_cluster','dropoff_cluster']
    # ]:
    #     prefix = "cluster_agg_" + prefix_sep.join(gby_cols)
    #     dropoff_cluster_agg_dict = {
    #         "arriveshop_cost": {prefix + "_arriveshop_cost_mean": "median",
    #                             prefix + "_arriveshop_cost_std": "std",
    #                             },
    #         "food_total_value": {
    #             prefix + "_food_total_value_sum": "sum"},
    #         "arrive_guest_cost": {prefix + "_aarrive_guest_cost_mean": "median",
    #                               prefix + "_arrive_guest_cost_std": "std"}
    #     }
    #
    #     tmp_d = nm.ka_add_groupby_features_1_vs_n(tr, gby_cols, dropoff_cluster_agg_dict)
    #     tr = pd.merge(tr, tmp_d, how='left', on=gby_cols)
    #     te = pd.merge(te, tmp_d, how='left', on=gby_cols)
    # poi 未来15分钟内的总配送价值, area 未来15分钟内的.
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
    #                      "cur_10min_deliverying_order_num_cha": lambda x: x.iloc[-1] - x.iloc[0]
    #                  }
    #                  }
    # cur_area_dict = {"_10min_working_rider_num": {"cur_10min_working_rider_last": "median"},
    #                  "_10min_notbusy_working_rider_num": {"cur_10min_notbusy_working_rider_median": "median"},
    #                  "_10min_not_fetched_order_num": {"cur_10min_not_fetched_order_num": "median"
    #                                                   },
    #                  "_10min_deliverying_order_num": {
    #                      "cur_10min_deliverying_order_num": "median"
    #                  }
    #                  }
    # area11 = nm.ka_add_groupby_features_1_vs_n(tr, ["area_id", 'cur_min'], cur_area_dict)
    # te_area_11 = nm.ka_add_groupby_features_1_vs_n(te, ["area_id", 'cur_min'], cur_area_dict)
    #
    # tr = pd.merge(tr, area11, how='left', left_on=["area_id", 'next15_min'], right_on=["area_id", "cur_min"])
    # te = pd.merge(te, te_area_11, how='left', left_on=["area_id", 'next15_min'], right_on=["area_id", "cur_min"])


    # gby_cols = ["poi_lat_bin", "poi_lng_bin"]
    #
    # tr = nm.get_rolling_target(tr, gby_cols, ["food_total_value", "delivery_distance"
    #     , "waiting_order_num"], prefix="rolling_poi_bin")
    # te = nm.get_rolling_target(te, gby_cols, ["food_total_value", "delivery_distance"
    #     , "waiting_order_num"], prefix="rolling_poi_bin")
    #
    # gby_cols = ["poi_id"]
    #
    # tr = nm.get_rolling_target(tr, gby_cols, ["food_total_value", "delivery_distance"
    #     , "waiting_order_num"], prefix="rolling_poi_fea")
    # te = nm.get_rolling_target(te, gby_cols, ["food_total_value", "delivery_distance"
    #     , "waiting_order_num"], prefix="rolling_poi_fea")
    prefix = "high_level_arrive_shop"
    gene_name = lambda p,x: "##".join([prefix,x])
    # tr[gene_name(prefix,"not_fetched_area_busy")] = (tr._10min_rain+1)*tr._10min_not_fetched_order_num/tr._10min_notbusy_working_rider_num
    tr[gene_name(prefix,"not_waited_div_not_working_rider")] = tr.waiting_order_num*(tr._10min_not_fetched_order_num + 1)

    # te[gene_name(prefix,"not_fetched_area_busy")] = (te._10min_rain+1)*te._10min_not_fetched_order_num/te._10min_notbusy_working_rider_num
    te[gene_name(prefix,"not_waited_div_not_working_rider")] = te.waiting_order_num*(te._10min_notbusy_working_rider_num + 1)

    prefix = "high_level_arrive_guest"
    tr[gene_name(prefix,"delivery_distance_1")] = (tr.delivery_distance/1000)*(tr._10min_not_fetched_order_num /(tr._10min_notbusy_working_rider_num+5))*(tr._10min_rain/2 +1.0)
    te[gene_name(prefix,"delivery_distance_1")] = (te.delivery_distance/1000)*(te._10min_not_fetched_order_num /(te._10min_notbusy_working_rider_num+5))*(te._10min_rain/2 +1.0)


    tr.to_csv(args.input_path + "train_sub.csv")
    te.to_csv(args.input_path + "test_sub.csv")
