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


def parse_to_min(x):
    hour = x[:2]
    min = x[-2:]
    return int(hour) * 60 + int(min)


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
    df.loc[mask, 'cur_min'] = df.loc[mask, c].apply(lambda x: parse_to_min(x.format("HHmm")))
    df.loc[mask, 'next5_min'] = df.loc[mask, c].apply(lambda x: parse_to_min(x.replace(minute=+15).format("HHmm")))
    df.loc[mask, 'next10_min'] = df.loc[mask, c].apply(lambda x: parse_to_min(x.replace(minute=+30).format("HHmm")))
    df.loc[mask, 'cur_10_block'] = df.loc[mask, c].apply(lambda x: (x.hour * 60 + x.minute) / 10)
    df.loc[mask, 'last_10_block'] = df.loc[mask, c].apply(lambda x: (x.hour * 60 + x.minute - 10) / 10)
    df.loc[mask, 'next_10_block'] = df.loc[mask, c].apply(lambda x: (x.hour * 60 + x.minute + 10) / 10)

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


def gene_mask(df, h, high=False, do_base=True):
    if do_base:
        base = (df.delivery_duration < 4654.0) & (df.delivery_duration > 663.0) & \
               (df.delivery_distance > 10)  # ???
    else:
        base = (df.delivery_duration > 0)

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

    ori.loc[:, 'poi_lat_bin1'] = np.round(ori['poi_lat'], 1)
    ori.loc[:, 'poi_lng_bin1'] = np.round(ori['poi_lng'], 1)
    ori.loc[:, 'cst_lat_bin1'] = np.round(ori['customer_latitude'], 1)
    ori.loc[:, 'cst_lng_bin1'] = np.round(ori['customer_longitude'], 1)

    ori.loc[:, 'poi_lat_bin'] = np.round(ori['poi_lat'], 2)
    ori.loc[:, 'poi_lng_bin'] = np.round(ori['poi_lng'], 2)
    ori.loc[:, 'cst_lat_bin'] = np.round(ori['customer_latitude'], 2)
    ori.loc[:, 'cst_lng_bin'] = np.round(ori['customer_longitude'], 2)

    ori.loc[:, 'poi_lat_bin3'] = np.round(ori['poi_lat'], 3)
    ori.loc[:, 'poi_lng_bin3'] = np.round(ori['poi_lng'], 3)
    ori.loc[:, 'cst_lat_bin3'] = np.round(ori['customer_latitude'], 3)
    ori.loc[:, 'cst_lng_bin3'] = np.round(ori['customer_longitude'], 3)

    ori.loc[:, 'poi_lat_bin4'] = np.round(ori['poi_lat'], 4)
    ori.loc[:, 'poi_lng_bin4'] = np.round(ori['poi_lng'], 4)
    ori.loc[:, 'cst_lat_bin4'] = np.round(ori['customer_latitude'], 4)
    ori.loc[:, 'cst_lng_bin4'] = np.round(ori['customer_longitude'], 4)

    # ori = ori.fillna(0)

    # tr_ori, te_ori = train_test_split(ori, test_size=0.5, random_state=2017)
    # tr_ori = tr_ori.loc[tr_ori.day != 30]
    # te_ori = te_ori.loc[(te_ori.day == 30) | (te_ori.day <= 22)]
    #
    # tr, pre = tr_ori[gene_mask(tr_ori, 11) | gene_mask(tr_ori, 12) | gene_mask(tr_ori, 13)], tr_ori[
    #     gene_mask(tr_ori, 10)]
    # te, ta = te_ori[gene_mask(te_ori, 11)], te_ori[gene_mask(te_ori, 10)]

    tr_ori = ori.loc[ori.day <= 16]
    te_ori = ori.loc[ori.day > 16]

    tr, pre = tr_ori[gene_mask(tr_ori, 11) | gene_mask(tr_ori, 12) | gene_mask(tr_ori, 13)], tr_ori[
        gene_mask(tr_ori, 10)]
    te, ta = te_ori[gene_mask(te_ori, 11, do_base=False)], te_ori[gene_mask(te_ori, 10, do_base=False)]

    # tr =
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


    print "poi_agg start"
    gby_cols = ["poi_id"]
    poi_agg = nm.get_stats_target(tr,
                                  gby_cols,
                                  ['arrive_guest_avg_speed', "avg_speed", "delivery_duration", "arriveshop_cost",
                                   "arrive_guest_cost", "fetch_cost",
                                   "delivery_distance", "food_total_value", "box_total_value",
                                   "waiting_order_num"],
                                  tgt_stats=["mean","median", "std"],
                                  drop_count=False,
                                  filter_count=15,
                                  prefix="poi_agg")
    tr = pd.merge(tr, poi_agg, how='left', left_on=gby_cols, right_on=gby_cols)
    te = pd.merge(te, poi_agg, how='left', left_on=gby_cols, right_on=gby_cols)

    print "lasthour poi_agg start"
    gby_cols = ["poi_id"]
    pre_poi_agg = nm.get_stats_target(pre,
                                      gby_cols,
                                      ['arrive_guest_avg_speed', "avg_speed", "delivery_duration", "arriveshop_cost",
                                       "arrive_guest_cost", "fetch_cost",
                                       "delivery_distance", "food_total_value", "box_total_value",
                                       "waiting_order_num"],
                                      tgt_stats=["mean","median", "std"],
                                      drop_count=True,
                                      filter_count=5,
                                      prefix="lasthour_poi_agg")
    ta_poi_agg = nm.get_stats_target(ta,
                                     gby_cols,
                                     ['arrive_guest_avg_speed', "avg_speed", "delivery_duration", "arriveshop_cost",
                                      "arrive_guest_cost", "fetch_cost",
                                      "delivery_distance", "food_total_value", "box_total_value",
                                      "waiting_order_num"],
                                     tgt_stats=["mean","median", "std"],
                                     drop_count=True,
                                     filter_count=5,
                                     prefix="lasthour_poi_agg")
    tr = pd.merge(tr, pre_poi_agg, how='left', left_on=gby_cols, right_on=gby_cols)
    te = pd.merge(te, ta_poi_agg, how='left', left_on=gby_cols, right_on=gby_cols)

    for gby_cols in [
        ['poi_id']
    ]:
        prefix = "lasthour_poi_agg_"
        lasthour_poi_last_dict = {
            "food_total_value": {
                "lasthour_poi_dict__food_total_value_last_ten": lambda x: np.mean(x.iloc[-5:-1])
            },
            "arriveshop_cost": {"lasthour_poi_dict__arriveshop_cost": lambda x: np.mean(x.iloc[-5:-1])},
            "fetch_cost": {"lasthour_poi_dict__fetch_cost": lambda x: np.mean(x.iloc[-5:-1])},
        }

        pre_poi_info_last_5 = nm.ka_add_groupby_features_1_vs_n(pre, gby_cols, lasthour_poi_last_dict)
        ta_poi_info_last_5 = nm.ka_add_groupby_features_1_vs_n(ta, gby_cols, lasthour_poi_last_dict)

        tr = pd.merge(tr, pre_poi_info_last_5, how='left', on=gby_cols)
        te = pd.merge(te, ta_poi_info_last_5, how='left', on=gby_cols)

    tr['prd_arrive_guest_time'] = tr['delivery_distance'] / tr['poi_agg__arrive_guest_avg_speed__median_by__poi_id']
    te['prd_arrive_guest_time'] = te['delivery_distance'] / te['poi_agg__arrive_guest_avg_speed__median_by__poi_id']

    print "area_agg start"
    gby_cols = ['area_id', 'hour', 'minute']
    area_agg_1 = nm.get_stats_target(tr,
                                     gby_cols,
                                     [
                                         'prd_arrive_guest_time',
                                         "arriveshop_cost"],
                                     tgt_stats=["median", "std"],
                                     drop_count=False,
                                     filter_count=10,
                                     prefix="area_agg")
    area_agg_2 = nm.get_stats_target(tr,
                                     gby_cols,
                                     ['food_total_value', "delivery_distance"],
                                     tgt_stats=["sum"],
                                     drop_count=True,
                                     filter_count=10,
                                     prefix="area_agg")

    tr = pd.merge(tr, area_agg_1, how='left', left_on=gby_cols, right_on=gby_cols)
    tr = pd.merge(tr, area_agg_2, how='left', left_on=gby_cols, right_on=gby_cols)
    te = pd.merge(te, area_agg_1, how='left', left_on=gby_cols, right_on=gby_cols)
    te = pd.merge(te, area_agg_2, how='left', left_on=gby_cols, right_on=gby_cols)

    print "pick_drop_cluster_agg start"

    for prefix, gby_cols in [
        # ("pick_cluster_agg", ["pickup_cluster"]),
        # ("drop_cluster_agg", ["dropoff_cluster"]),
        ("user_bin2", ["cst_lat_bin", "cst_lng_bin"]),
        ("poi_bin2", ["poi_lat_bin", "poi_lng_bin"]),
        ("15min_user_bin", ["cst_lat_bin", "cst_lng_bin", "15min"]),
        ("15min_poi_bin", ["poi_lat_bin", "poi_lng_bin", "15min"]),
    ]:
        agg = nm.get_stats_target(tr,
                                  gby_cols,
                                  ["delivery_duration", "arriveshop_cost",
                                   "arrive_guest_cost", "fetch_cost",
                                   "delivery_distance", "food_total_value", "box_total_value",
                                   "waiting_order_num", "area_busy_coef", "_10min_deliverying_order_num",
                                   "_10min_not_fetched_order_num"],
                                  tgt_stats=["median", "std", "sum", "max", "min"],
                                  drop_count=True,
                                  filter_count=10,
                                  prefix=prefix)
        tr = pd.merge(tr, agg, how='left', left_on=gby_cols, right_on=gby_cols)
        te = pd.merge(te, agg, how='left', left_on=gby_cols, right_on=gby_cols)

    for prefix, gby_cols in [
        ("user_bin3", ["cst_lat_bin3", "cst_lng_bin3"]),
        ("poi_bin3", ["poi_lat_bin3", "poi_lng_bin3"]),
        ("poi_user_bin2", ["cst_lat_bin", "cst_lng_bin", "poi_lat_bin", "poi_lng_bin"]),
        # ("poi_user_bin3", ["cst_lat_bin", "cst_lng_bin", "poi_lat_bin3", "poi_lng_bin3"]),
        # ("user_bin4", ["cst_lat_bin4", "cst_lng_bin4"]),
        # ("poi_bin4", ["poi_lat_bin4", "poi_lng_bin4"]),
        # ("poi_user_bin4", ["cst_lat_bin3", "cst_lng_bin3", "poi_lat_bin4", "poi_lng_bin4"]),
        # ("pick_drop_cluster_agg", ["pickup_cluster", "dropoff_cluster"]),
        ("poi_user_bin", ["cst_lat_bin", "cst_lng_bin", "poi_lat_bin", "poi_lng_bin"]),
        # ("15min_pick_drop_cluster_agg", ["pickup_cluster", "dropoff_cluster", "15min"]),

    ]:
        agg = nm.get_stats_target(tr,
                                  gby_cols,
                                  ["delivery_duration", "arriveshop_cost",
                                   "arrive_guest_cost", "fetch_cost",
                                   "delivery_distance", "food_total_value",
                                   "waiting_order_num"],
                                  tgt_stats=["median", "std", "sum", "max", "min"],
                                  drop_count=True,
                                  filter_count=10,
                                  prefix=prefix)
        tr = pd.merge(tr, agg, how='left', left_on=gby_cols, right_on=gby_cols)
        te = pd.merge(te, agg, how='left', left_on=gby_cols, right_on=gby_cols)

    gby_cols = ["area_id", 'cur_10_block']
    area11 = nm.get_stats_target(tr,
                                 gby_cols,
                                 ["_10min_not_fetched_order_num", "food_total_value",
                                  "delivery_distance", ],
                                 tgt_stats=["sum"],
                                 drop_count=False,
                                 filter_count=5,
                                 prefix="cur_block_area_info")
    te_area_11 = nm.get_stats_target(te,
                                     gby_cols,
                                     ["_10min_not_fetched_order_num", "food_total_value",
                                      "delivery_distance", ],
                                     tgt_stats=["sum"],
                                     drop_count=False,
                                     filter_count=5,
                                     prefix="cur_block_area_info")

    gby_cols = ["poi_id", 'cur_10_block']
    poi_11 = nm.get_stats_target(tr,
                                 gby_cols,
                                 ["food_total_value", "delivery_distance",
                                  "box_total_value", "waiting_order_num"],
                                 tgt_stats=["sum", "mean", "std"],
                                 drop_count=False,
                                 filter_count=1,
                                 prefix="cur_block_poi_info")
    te_poi_11 = nm.get_stats_target(te,
                                    gby_cols,
                                    ["food_total_value", "delivery_distance",
                                     "box_total_value", "waiting_order_num"],
                                    tgt_stats=["sum", "mean", "std"],
                                    drop_count=False,
                                    filter_count=1,
                                    prefix="cur_block_poi_info")

    tr = pd.merge(tr, area11.set_index(["area_id", 'cur_10_block']), how='left', left_on=["area_id", 'next_10_block'],
                  right_index=True)
    te = pd.merge(te, te_area_11.set_index(["area_id", 'cur_10_block']), how='left',
                  left_on=["area_id", 'next_10_block'], right_index=True)
    tr = pd.merge(tr, area11.set_index(["area_id", 'cur_10_block']), how='left', left_on=["area_id", 'last_10_block'],
                  right_index=True)
    te = pd.merge(te, te_area_11.set_index(["area_id", 'cur_10_block']), how='left',
                  left_on=["area_id", 'last_10_block'], right_index=True)
    del area11
    del te_area_11

    tr = pd.merge(tr, poi_11.set_index(["poi_id", 'cur_10_block']), how='left', left_on=["poi_id", 'next_10_block'],
                  right_index=True)
    te = pd.merge(te, te_poi_11.set_index(["poi_id", 'cur_10_block']), how='left', left_on=["poi_id", 'next_10_block'],
                  right_index=True)
    tr = pd.merge(tr, poi_11.set_index(["poi_id", 'cur_10_block']), how='left', left_on=["poi_id", 'last_10_block'],
                  right_index=True)
    te = pd.merge(te, te_poi_11.set_index(["poi_id", 'cur_10_block']), how='left', left_on=["poi_id", 'last_10_block'],
                  right_index=True)
    tr = pd.merge(tr, poi_11.set_index(["poi_id", 'cur_10_block']), how='left', left_on=["poi_id", 'cur_10_block'],
                  right_index=True)
    te = pd.merge(te, te_poi_11.set_index(["poi_id", 'cur_10_block']), how='left', left_on=["poi_id", 'cur_10_block'],
                  right_index=True)

    del poi_11
    del te_poi_11
    gby_cols = ["cst_lat_bin", "cst_lng_bin"]

    tr = nm.get_rolling_target(tr, gby_cols, ["area_busy_coef", "_10min_deliverying_order_num",
                                              "_10min_not_fetched_order_num", "food_total_value", "delivery_distance"
        , "waiting_order_num"], prefix="rolling_user_bin")
    te = nm.get_rolling_target(te, gby_cols, ["area_busy_coef", "_10min_deliverying_order_num",
                                              "_10min_not_fetched_order_num", "food_total_value", "delivery_distance"
        , "waiting_order_num"], prefix="rolling_user_bin")

    gby_cols = ["poi_lat_bin", "poi_lng_bin"]

    tr = nm.get_rolling_target(tr, gby_cols, ["area_busy_coef", "_10min_deliverying_order_num",
                                              "_10min_not_fetched_order_num", "food_total_value", "delivery_distance"
        , "waiting_order_num"], prefix="rolling_poi_bin")
    te = nm.get_rolling_target(te, gby_cols, ["area_busy_coef", "_10min_deliverying_order_num",
                                              "_10min_not_fetched_order_num", "food_total_value", "delivery_distance"
        , "waiting_order_num"], prefix="rolling_poi_bin")

    gby_cols = ["poi_id"]

    tr = nm.get_rolling_target(tr, gby_cols, ["area_busy_coef", "_10min_deliverying_order_num",
                                              "_10min_not_fetched_order_num", "food_total_value", "delivery_distance"
        , "waiting_order_num"], prefix="rolling_poi_fea")
    te = nm.get_rolling_target(te, gby_cols, ["area_busy_coef", "_10min_deliverying_order_num",
                                              "_10min_not_fetched_order_num", "food_total_value", "delivery_distance"
        , "waiting_order_num"], prefix="rolling_poi_fea")


    prefix = "high_level_arrive_shop"
    gene_name = lambda p,x: "##".join([prefix,x])
    tr[gene_name(prefix,"not_fetched_area_busy")] = (tr._10min_rain+1)*tr._10min_not_fetched_order_num/tr._10min_notbusy_working_rider_num
    tr[gene_name(prefix,"not_waited_div_not_working_rider")] = tr.waiting_order_num/(tr._10min_notbusy_working_rider_num + 1)
    tr[gene_name(prefix,"waiting_plus_food_value_plus_box_value")] = 7*tr.waiting_order_num + tr.food_total_value + tr.box_total_value*4 + tr.food_num
    tr[gene_name(prefix,"waiting_mul_poi_agg__food_total_value__mean_by__poi_id")] = tr.waiting_order_num* tr.rolling_poi_fea__food_total_value__forward_5_mean__poi_id
    # tr[gene_name(prefix,"5")] = tr.food_total_value/tr.food_num

    te[gene_name(prefix,"not_fetched_area_busy")] = (te._10min_rain+1)*te._10min_not_fetched_order_num/te._10min_notbusy_working_rider_num
    te[gene_name(prefix,"not_waited_div_not_working_rider")] = te.waiting_order_num/(te._10min_notbusy_working_rider_num + 1)
    te[gene_name(prefix,"waiting_plus_food_value_plus_box_value")] = 7*te.waiting_order_num + te.food_total_value + te.box_total_value*4 + te.food_num
    te[gene_name(prefix,"waiting_mul_poi_agg__food_total_value__mean_by__poi_id")] = te.waiting_order_num* te.rolling_poi_fea__food_total_value__forward_5_mean__poi_id
    # te[gene_name(prefix,"5")] = te.food_total_value/te.food_num

    prefix = "high_level_arrive_guest"
    tr[gene_name(prefix,"delivery_distance_1")] = (tr.delivery_distance/100)*(tr._10min_not_fetched_order_num /(tr._10min_notbusy_working_rider_num+5))*(tr._10min_rain/2 +1.0)
    tr[gene_name(prefix,"delivery_distance_2")] = (tr.delivery_distance/100)*(tr.waiting_order_num/5 + 1)
    tr[gene_name(prefix,"delivery_distance_3")] = (tr.delivery_distance/100)/ (tr._10min_not_fetched_order_num + tr._10min_deliverying_order_num)
    tr[gene_name(prefix,"delivery_distance_4")] = tr.delivery_distance/tr._10min_deliverying_order_num
    tr[gene_name(prefix,"delivery_distance_5")] = tr.delivery_distance/ (tr._10min_notbusy_working_rider_num+1)

    te[gene_name(prefix,"delivery_distance_1")] = (te.delivery_distance/100)*(te._10min_not_fetched_order_num /(te._10min_notbusy_working_rider_num+5))*(te._10min_rain/2 +1.0)
    te[gene_name(prefix,"delivery_distance_2")] = (te.delivery_distance/100)*(te.waiting_order_num/5 + 1)
    te[gene_name(prefix,"delivery_distance_3")] = (te.delivery_distance/100)/ (te._10min_not_fetched_order_num + te._10min_deliverying_order_num)
    te[gene_name(prefix,"delivery_distance_4")] = te.delivery_distance/te._10min_deliverying_order_num
    te[gene_name(prefix,"delivery_distance_5")] = te.delivery_distance/ (te._10min_notbusy_working_rider_num+1)



    '''
    订单
    0.01 相关性的作为辅助特征.
    到店:

    (_10min_deliverying_order_num + _10min_not_fetched_order_num) / (_10min_working_rider_num + _10min_notbusy_working_rider_num)
    (_10min_rain + 1) *  _10min_not_fetched_order_num/_10min_notbusy_working_rider_num
    (_10min_rain + 1) * _10min_not_fetched_order_num/(_10min_working_rider_num + _10min_notbusy_working_rider_num)
    waiting_order_num/_10min_notbusy_working_rider_num
    waiting_order_num/(_10min_working_rider_num + _10min_notbusy_working_rider_num)
    waiting_order_num * (_10min_not_fetched_order_num + _10min_deliverying_order_num)/100
    (_10min_working_rider_num + _10min_notbusy_working_rider_num)
    2*waiting_order_num + food_total_value + box_total_value*4
    waiting_order_num* poi_agg__food_total_value__mean_by__poi_id
    food_total_value/(_10min_working_rider_num + _10min_notbusy_working_rider_num)

    到客:

    配送难度:
    配送距离/100 * _10min_not_fetched_order_num /10min_notbusy_working_rider_num * (下雨/2 + 1.0)
    配送距离/100 + waiting_order_number
    distance/_10min_deliverying_order_num
    distance/(_10min_deliverying_order_num + _10min_not_fetched_order_num)

    历史相关:

    出餐难度: (下雨 + 1) * (食物价值  + _10min_not_fetched_order_num/_10min_notbusy_working_rider_num - waiting_order_num )
    _10min_temperature????

    区域:
    poi:
    food_total_value *

    area  working order number /每分钟总单量. 区域的忙碌程度.
   '''


    tr.to_csv(args.output_path + "train_cv.csv")
    te.to_csv(args.output_path + "test_cv.csv")
