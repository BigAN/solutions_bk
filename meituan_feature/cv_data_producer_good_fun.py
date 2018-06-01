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
import math
# print(check_output(["ls", "./data"]).decode("utf8"))


# Any results you write to the current directory are saved as output.


from subprocess import check_output
import datetime
import arrow as ar
import pandas as pd
import numpy as np

# print(check_output(["ls", "./data"]).decode("utf8"))

prefix_sep = "#"
big_sep = "__"
get_key = lambda prefix,kk: big_sep.join([prefix, kk])


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


    def gene_poi_agg_dict(prefix):
        return {
            'arrive_guest_avg_speed': {"_arrive_guest_avg_speed_mean": "median",
                                       "_arrive_guest_avg_speed_std": "std"},
            'avg_speed': {"_avg_speed_mean" : "median",
                           "_avg_speed_std" : "std"},
            'delivery_duration': {'_mean_dd': 'median',
                                  '_std_dd': 'std'},
            "arriveshop_cost": {"_mean_arriveshop_cost": "median",
                                "_std_arriveshop_cost": "std"},
            "fetch_cost": {"_mean_fetch_cost": "median",
                           "_std_fetch_cost": "std",
                           },
            "arrive_guest_cost": {"_mean_arrive_guest_cost": "median",
                                  "_std_arrive_guest_cost": "std"},
            "delivery_distance": {"_mean_delivery_distance": "median",
                                  "_mean_delivery_distance": "std"},
            "food_total_value": {"_mean_food_total_value": "median",
                                 "_mean_food_total_value": "std"},
            "box_total_value": {"_mean_box_total_value": "median"},
            # "food_num": {"_mean_food_num": "median",
            #              "_total_food_num": "sum"},
            "waiting_order_num": {"_mean_waiting_order_num": "median"},
            "order_id": {"_coor_count": "count",
                         }
        }


    def deco_key(two_layer_dict,prefix):
        for k, v in two_layer_dict.items():
            for kk, vv in v.items():
                two_layer_dict[k][big_sep.join([prefix, kk])] = two_layer_dict[k].pop(kk)
        return two_layer_dict


    for gby_cols in [['poi_id'],['poi_id','15min']]:
        prefix = "poi_agg_" + prefix_sep.join(gby_cols)
        poi_agg_dict = deco_key(gene_poi_agg_dict(prefix),prefix)

        tmp_d = nm.ka_add_groupby_features_1_vs_n(tr, gby_cols, poi_agg_dict)
        coord_stats = tmp_d[tmp_d[get_key(prefix,"_coor_count")] > 10]
        tr = pd.merge(tr, coord_stats, how='left', on=gby_cols)
        te = pd.merge(te, coord_stats, how='left', on=gby_cols)

    for gby_cols in [
        ['area_id', 'hour', 'minute']
    ]:
        prefix = "area_id_"+ prefix_sep.join(gby_cols)
        area_agg_dict = {
            "arriveshop_cost": {"_arriveshop_cost_mean": "median",
                                "_arriveshop_cost_std": "std",
                                },
            "food_total_value": {
                "_food_total_value_sum": "sum"},
        }
        area_agg_dict= deco_key(area_agg_dict,prefix)

        tmp_d = nm.ka_add_groupby_features_1_vs_n(tr, gby_cols, area_agg_dict)
        tr = pd.merge(tr, tmp_d, how='left', on=gby_cols)
        te = pd.merge(te, tmp_d, how='left', on=gby_cols)

    # last hour poi  feature
    for gby_cols in [
        ['poi_id']
    ]:
        prefix = "lasthour_agg_" + prefix_sep.join(gby_cols)
        lasthour_poi_agg_dict = {
            'arrive_guest_avg_speed': {"arrive_guest_avg_speed_mean": "median",
                                       "_arrive_guest_avg_speed_std": "std"},
            'avg_speed': {"_avg_speed_mean": "median",
                          "_avg_speed_std": "std"},
            'delivery_duration': {'_mean_dd': 'median',
                                  '_std_dd': 'std'},
            "arriveshop_cost": {"_mean_arriveshop_cost": "median",
                                "_std_arriveshop_cost": "std"},
            "fetch_cost": {"_mean_fetch_cost": "median",
                           "_std_fetch_cost": "std",
                           },
            "arrive_guest_cost": {"_mean_arrive_guest_cost": "median",
                                  "_std_arrive_guest_cost": "std"},
            "delivery_distance": {"_mean_delivery_distance": "median",
                                  "_mean_delivery_distance": "std"},
            "food_total_value": {"_mean_food_total_value": "median",
                                 "_mean_food_total_value": "std"},
            "box_total_value": {"_mean_box_total_value": "median"},
            "waiting_order_num": {"_mean_waiting_order_num": "median"},
            "order_id": {"_coor_count": "count",
                         }
        }
        lasthour_poi_agg_dict = deco_key(lasthour_poi_agg_dict,prefix)

        pre_poi_info = nm.ka_add_groupby_features_1_vs_n(pre, gby_cols, lasthour_poi_agg_dict)
        ta_poi_info = nm.ka_add_groupby_features_1_vs_n(ta, gby_cols, lasthour_poi_agg_dict)
        tr = pd.merge(tr, pre_poi_info, how='left', on=gby_cols)
        te = pd.merge(te, ta_poi_info, how='left', on=gby_cols)

    tr['prd_arrive_guest_time'] = tr['delivery_distance'] / tr[get_key("poi_agg_" + prefix_sep.join(gby_cols),"_arrive_guest_avg_speed_mean")]
    te['prd_arrive_guest_time'] = te['delivery_distance'] / te[get_key("poi_agg_" + prefix_sep.join(gby_cols),"_arrive_guest_avg_speed_mean")]

    tr.to_csv(args.input_path + "train_cv.csv")
    te.to_csv(args.input_path + "test_cv.csv")
    '''
       应该有效果的方向
        3. 趋势特征, last hour 分为 30 60. 2. 当前的过去15min,30min 的均值.
        # ["poi_lat_bin",
        #  "poi_lng_bin",
        #  "cst_lat_bin",
        #  "cst_lng_bin"]
        4. 分析特征相关性.
    '''
