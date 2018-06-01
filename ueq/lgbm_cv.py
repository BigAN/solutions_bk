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
import numpy as np
import operator

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('You need to install matplotlib for plot_example.py.')

from sklearn.metrics import mean_absolute_error

cate_features = ["poi_id", "area_id", "15min", "10min", "direction", "poi_lat_bin", "poi_lng_bin", "cst_lat_bin",
                 "cst_lng_bin"]
# id	uuid	wm_poi_id	poi_id	city_id	shipping_fee	order_time	user_id	total	original_price	is_donation	is_discount_fee	longitude	latitude	wk	mo	reorder	poi_reorder_rate	poi_reorder_counts	poi_one_bought	poi_two_bought	user_reorder_rate	user_reorder_count	dt

to_drop = [
    "uuid", "poi_id", "wm_poi_id", "reorder", "dt", "order_time", "user_id", "wk", 'mo', 'city_id', "is_discount_fee",
    "is_donation", "shipping_fee", "id", "total", "original_price", "longitude", "latitude", "timescope_order_cnt",
    "timescope_total_price", "timescope_original_price","user_poi_order_decay"
    # ,"user_reorder_rate","user_reorder_count"
]

# drop 85
# fea_drop =  [u'order_cnt_5hr', u'order_cnt_3hr', u'order_cnt_2hr', u'order_cnt_4hr', u'order_cnt_8hr', u'original_price_7day', u'order_cnt_6hr', u'comment_2star', u'service_fee_rate_7day', u'order_cnt_7day', u'order_cnt_1hr', u'order_cnt_23hr', u'order_cnt_increase', u'order_cnt_7hr', u'order_cnt_9hr', u'order_cnt_24hr', u'order_cnt_15hr', u'comment_1star', u'order_cnt_22hr', u'order_cnt_19hr', u'order_cnt_12hr', u'order_cnt_14hr', u'pv_cxr_1day', u'order_cnt_17hr', u'order_cnt_21hr', u'order_cnt_18hr', u'order_cnt_16hr', u'pic_comment_cnt', u'uv_ctr_1day', u'avg_food_comment_score', u'uv_cvr_15day', u'comment_4star', u'month_original_price', u'user_reorder_count', u'click_tag_pref_3day', u'dp_score', u'month_order_cnt', u'month_total_price', u'uv_cxr_7day', u'op_time_30day', u'click_tag_pref_30day', u'click_tag_pref_15day', u'pv_ctr_7day', u'uv_cvr_7day', u'distance_30day', u'order_cnt_13hr', u'pv_ctr_15day', u'uv_cvr_1day', u'uv_ctr_15day', u'service_fee_7day', u'pv_ctr_1day', u'discount_rate_all_customer_7day', u'food_comment_rate', u'pos_delivery_comment_rate', u'avg_delivery_comment_score', u'dp_avg_price', u'pv_cvr_1day', u'order_cnt_20hr', u'pos_comment_rate', u'click_tag_pref_7day', u'avg_price_month', u'area_30day', u'food_comment_cnt', u'poi_one_bought',
#              u'comment_3star',
#              u'uv_ctr_7day', u'uv_cxr_1day', u'service_fee_1day',
#              u'comment_5star',
#              u'comment_5star_rate', u'pic_comment_rate', u'avg_comment_score', u'order_cnt_11hr',
#              u'total_price_1day',
#              # "order_cnt_1day",
#              u'original_price_1day', u'discount_rate_new_customer_7day', u'uv_cxr_15day', u'order_cnt_14day', u'tag_id', u'neg_delivery_comment_rate', u'pv_cxr_7day', u'comment_uv', u'pv_cvr_7day', u'poi_two_bought', u'poi_reorder_counts']
# features = list((set(basic_features) | set(advanced_features)) - set(to_drop) - set(fea_drop))
#drop 45
fea_drop =["tag_id"]


# features = single_feas

best_feas = [u'poi_reorder_rate', u'poi_reorder_counts',
             u'poi_one_bought', u'poi_two_bought', u'user_reorder_rate',
             u'user_reorder_count']


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


import random


def deco_data(feas, is_train=False):
    feas.columns = map(lambda x: x.replace("waimai_ad_join_reorder_v8.", ""), feas.columns.tolist())
    # rs = nm.ka_add_groupby_features_1_vs_n(feas, ["uuid"], {"reorder": {"sum": "sum", "count": "count"}})
    if is_train:
        # with_reorder = pd.DataFrame(rs[rs['sum'] > 0]["uuid"])
        # feas = pd.merge(feas, with_reorder, how="inner", on="uuid")
        b_len = len(feas)
        # feas = feas[feas['user_reorder_rate'] > 0]
        print "before filter , lenght is {}, afther filter length is {}".format(b_len, len(feas))

    # feas['poi_reorder_rate'] = feas["user_reorder_rate"]*3 + feas["poi_reorder_rate"]
    # feas['poi_reorder_counts'] = feas["user_reorder_count"]*10 + feas["poi_reorder_counts"]
    # feas['poi_reorder_rate'] = feas["poi_reorder_rate"].apply(lambda x: round(x, 3))

    return feas


if __name__ == '__main__':
    args = init_arguments()

    train_data = deco_data(pd.read_csv(args.train_path, sep="\t", skiprows=1))
    test_data = deco_data(pd.read_csv(args.test_path, sep="\t", skiprows=1))
    print train_data.head(5)
    print train_data.dtypes
    features = list(
            set(train_data.columns.tolist()[1:]) - set(
                    to_drop) - set(fea_drop))

    print "len feas {}, feas {}".format(len(features), features)
    # features = best_feas
    # train_data.fillna(-999, inplace=True)
    # test_data.fillna(-999, inplace=True)

    labels = train_data[args.label].values.astype(np.float32).flatten()
    labels_val = test_data[[args.label]].values.astype(np.float32).flatten()
    lgb_train = lgb.Dataset(train_data[features], labels)
    lgb_eval = lgb.Dataset(test_data[features], labels_val, reference=lgb_train)

    print('Start training...')

    params = {'num_leaves': 100, 'task': 'train', 'verbose': 1, 'learning_rate': 0.03, 'nthread': 8,
              'min_data_in_leaf': 150, 'objective': 'binary', 'boosting_type': 'gbdt', 'metric': ['logloss', 'auc'],
              'feature_fraction': 0.1, "bagging_fraction": 0.7, "bagging_freq": 5}

    gbm = lgb.train(params,
                    lgb_train,
                    # fobj=huber_approx_obj,
                    valid_sets=[lgb_eval],
                    num_boost_round=args.round,
                    early_stopping_rounds=30,
                    verbose_eval=10)

    df = pd.DataFrame({'feature': gbm.feature_name(), 'importances': gbm.feature_importance()})
    df['fscore'] = df['importances'] / df['importances'].sum()

    # print(df)
    print(df.sort_values('importances', ascending=False).to_string(index=False))
    print  "-" * 20 + " features" + "-" * 20
    print [x.strip() for x in df.sort_values('importances', ascending=False).feature.to_string(index=False).split("\n")]
    # print(df[df['fscore']<=0.005].feature.to_string(index=False))

    test_data['prd'] = gbm.predict(test_data[features])
    import os

    # ax = lgb.plot_tree(gbm,tree_index=8, figsize=(20, 8), show_info=['split_gain'])
    plt.show()
    test_data.to_csv(args.out_path)
