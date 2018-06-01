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

features = [u'pic_comment_cnt',
            u'original_price_7day',
            u'pic_comment_rate',
            u'service_fee_rate_7day',
            u'pv_ctr_7day',
            u'uv_cxr_7day',
            u'neg_delivery_comment_rate',
            u'pos_comment_rate',
            u'comment_5star',
            u'month_original_price',
            u'dp_avg_price',
            u'month_order_cnt',
            u'order_cnt_14hr',
            u'order_cnt_3hr',
            u'click_tag_pref_3day',
            u'comment_3star',
            u'order_cnt_14day',
            u'total_price_1day',
            u'pv_ctr_15day',
            u'click_tag_pref_15day',
            u'uv_ctr_15day',
            u'avg_price_month',
            u'uv_cvr_15day',
            u'order_cnt_21hr',
            u'order_cnt_17hr',
            u'order_cnt_increase',
            u'order_cnt_4hr',
            u'service_fee_7day',
            u'avg_food_comment_score',
            u'order_cnt_15hr',
            u'uv_cxr_15day',
            u'order_cnt_13hr',
            u'pos_delivery_comment_rate',
            u'food_comment_rate',
            u'order_cnt_7hr',
            u'area_30day',
            u'poi_one_bought',
            u'pv_cvr_7day',
            u'comment_1star',
            u'original_price_1day',
            u'user_reorder_count',
            u'food_comment_cnt',
            u'order_cnt_16hr',
            u'discount_rate_new_customer_7day',
            u'uv_cxr_1day',
            u'discount_rate_all_customer_7day',
            u'order_cnt_12hr',
            u'order_cnt_19hr',
            u'order_cnt_23hr',
            u'order_cnt_20hr',
            u'order_cnt_22hr',
            u'uv_cvr_1day',
            u'order_cnt_9hr',
            u'order_cnt_2hr',
            u'order_cnt_24hr',
            u'order_cnt_5hr',
            u'avg_delivery_comment_score',
            u'click_tag_pref_30day',
            u'comment_uv',
            u'dp_score',
            u'order_cnt_18hr',
            u'poi_two_bought',
            u'order_cnt_7day',
            u'uv_ctr_7day',
            u'order_cnt_11hr',
            u'order_cnt_6hr',
            u'month_total_price',
            u'service_fee_1day',
            u'pv_cvr_1day',
            u'distance_30day',
            u'comment_5star_rate',
            u'order_cnt_1hr',
            u'order_cnt_8hr',
            u'pv_cxr_1day',
            u'comment_4star',
            u'tag_id',
            u'pv_cxr_7day',
            u'pv_ctr_1day',
            u'comment_2star',
            u'avg_comment_score',
            u'click_tag_pref_7day',
            u'uv_cvr_7day',
            u'uv_ctr_1day',
            u'op_time_30day']

base = [u'order_tag_pref_3day',
        u'order_cnt_10hr',
        u'total_price_7day',
        u'user_reorder_rate',
        u'order_cnt_1day',
        u'neg_comment_rate',
        u'order_tag_pref_7day',
        u'submit_tag_pref_30day',
        u'service_fee_rate_1day',
        u'submit_tag_pref_3day',
        u'pv_cxr_15day',
        u'order_tag_pref_15day',
        u'user_poi_submit_decay',
        u'pv_cvr_15day',
        u'user_poi_order_decay',
        u'submit_tag_pref_7day',
        u'poi_reorder_rate',
        u'submit_tag_pref_15day',
        u'user_poi_click_decay', 'order_tag_pref_30day']


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

log= []
if __name__ == '__main__':
    args = init_arguments()
    train_data = pd.read_csv(args.train_path, sep="\t", skiprows=1)
    test_data = pd.read_csv(args.test_path, sep="\t", skiprows=1)

    labels = train_data[[args.label]].values.astype(np.float32).flatten()
    labels_val = test_data[[args.label]].values.astype(np.float32).flatten()

    rs = []
    score = 0
    best_drop = []
    init_feas = base
    for f in [None]+features:
        print "try with f {}".format(f)
        tmp_features = init_feas + [f] if f else init_feas
        print tmp_features
        # se_features = list(set(tmp_features) - set([f]) - set(best_drop))
        lgb_train = lgb.Dataset(train_data[tmp_features], labels)
        lgb_eval = lgb.Dataset(test_data[tmp_features], labels_val, reference=lgb_train)

        params = {'num_leaves': 100, 'task': 'train', 'verbose': 1, 'learning_rate': 0.01, 'nthread': 8,
                  'min_data_in_leaf': 10, 'objective': 'binary', 'boosting_type': 'gbdt', 'metric': ['auc'],
                  'feature_fraction': 0.15, "bagging_fraction": 0.7, "bagging_freq": 5}

        print('Start training...')
        # train
        gbm = lgb.train(params,
                        lgb_train,
                        # fobj=huber_approx_obj,
                        valid_sets=[lgb_eval],
                        num_boost_round=args.round,
                        early_stopping_rounds=30,
                        verbose_eval=10)

        # print('Feature names:', gbm.feature_name())

        # print('Calculate feature importances...')
        # feature importances
        # print('Feature importances:', list(gbm.feature_importance()))

        df = pd.DataFrame({'feature': gbm.feature_name(), 'importances': gbm.feature_importance()})
        # print(df.sort_values('importances'))
        test_data.loc[:, "delivery_duration_prd"] = gbm.predict(test_data[tmp_features])
        print gbm.best_score
        cur_s = gbm.best_score['valid_0']['auc']

        print "cur f is {},best score is {}, cur_s is {}, ".format(f, score, cur_s)
        if f and (score == 0 or cur_s > score):
            score = cur_s
            print "best add f {}".format(f)
            init_feas.append(f)
        log.append([f,score,cur_s])


    print "final score is  {},init_feas is {}".format(score, init_feas)
    print log
    # rs = order_id.merge(test_data[["order_id", args.label]], left_on="order_id", right_on="order_id",
    #                     how="left")
    #
    # rs.to_csv(args.out_path, header=['order_id', args.label], index=False)
    # print sorted(rs, key=lambda x: -x[1])
    # print "\n".join(map(lambda x: ":".join(map(str, x)), sorted(rs, key=lambda x: -x[1])))
