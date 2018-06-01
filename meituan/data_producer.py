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
    df.loc[mask, 'hour'] = df.loc[matsk, c].apply(lambda x: x.hour)
    df.loc[mask, 'minute'] = df.loc[mask, c].apply(lambda x: x.minute)
    df.loc[mask, c] = df.loc[mask, "tmp_order_unix_time"]

    return df


def load_area_data(file_name):
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
    weather_realtime = load_area_data(path_data + 'weather_realtime.csv')
    waybill_info_test_a = load_order_data(path_data + 'waybill_info_test_a.csv', sep="\t")
    waybill_info_test_b = load_order_data(path_data + 'waybill_info_test_b.csv')
    area_realtime_test = load_area_data(path_data + 'area_realtime_test.csv')
    weather_realtime_test = load_area_data(path_data + 'weather_realtime_test.csv')

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
    te = waybill_info_test_b.merge(weather_realtime_test, on=['date', 'hour', 'minute', 'area_id'], how='left') \
        .merge(area_realtime_test, on=['date', 'hour', 'minute', 'area_id'], how='left')

    print "tr {},te {}".format(len(tr),len(te))
    tr.to_csv(args.input_path + "train.csv")
    te.to_csv(args.input_path + "test.csv")
