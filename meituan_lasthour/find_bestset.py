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

features = [x[0] for x in [['food_total_value', 404.75606428212961], ['area_busy_coef', 403.08878638362586],
            ['waiting_order_num', 401.69546250388026], ['lasthour_poi_agg_poi_id_mean_dd', 400.999701241293],
            ['poi_agg_poi_id_mean_waiting_order_num', 400.9843086673601],
            ['poi_agg_poi_id_mean_fetch_cost', 400.97838397222802], ['poi_agg_poi_id_std_dd', 400.97765098664843],
            ['cur_order_id_count_y', 400.96963327287557],
            ['lasthour_poi_agg__avg_speed_meanpoi_id', 400.84792903222365], ['10min', 400.78581906043235],
            ['lasthour_poi_dict__fetch_cost', 400.78156195992597], ['cur_block_box_total_value_y', 400.77969941545251],
            ['cst_lat_bin', 400.77771760254916], ['lasthour_poi_dict__food_total_value_last_ten', 400.77461129378673],
            ['poi_agg_poi_id_mean_arriveshop_cost', 400.76538007374086],
            ['lasthour_poi_agg_poi_id_coor_count', 400.76496165230219],
            ['poi_agg_poi_id_std_food_total_value', 400.76153921907576],
            ['future_delivery_distance_x', 400.75878330455066],
                           # ['all_features', 400.74165719519806],
            ['poi_agg_poi_id_std_fetch_cost', 400.72258678443683], ['area_id', 400.71499897367801],
            ['lasthour_poi_agg_poi_id_mean_food_total_value', 400.70695821549856],
            ['customer_latitude', 400.69921433825431], ['future_delivery_distance_y', 400.69420454351979],
            ['poi_lat_bin', 400.66974428839126], ['cur_order_id_count_x', 400.65965804873213],
            ['bill_number_per_rider', 400.64568323048263], ['poi_id', 400.64522042284585],
            ['area_id_area_id#hour#minute_prd_arrive_guest_time_std', 400.64127791154255],
            ['prd_arrive_guest_time', 400.63558712332645], ['cur_10min_not_fetched_order_num_x', 400.62974143099513],
            ['poi_agg_poi_id_std_arriveshop_cost', 400.62338137855176],
            ['lasthour_poi_agg_poi_id_std_dd', 400.62181116200463],
            ['_10min_deliverying_order_num', 400.61758488185444], ['cur_block_box_total_value', 400.60296128060924],
            ['future_food_total_value_sum_y', 400.59626376714789], ['delivery_distance', 400.5898525418242],
            ['cur_block_delivery_distance', 400.57453489516735],
            ['lasthour_poi_agg_poi_id_mean_arriveshop_cost', 400.56464945397636], ['poi_lat', 400.56008371025052],
            ['cur_area_busy_coef_x', 400.54738921337514], ['lasthour_poi_dict__arriveshop_cost', 400.54209948378804],
            ['cur_waiting_order_num_x', 400.52561711964404], ['poi_agg__avg_speed_meanpoi_id', 400.52135261687999],
            ['lasthour_poi_agg_poi_id_std_arrive_guest_cost', 400.51481207668297],
            ['lasthour_poi_agg_poi_id_arrive_guest_avg_speed_std', 400.51366525174438],
            ['lasthour_poi_agg_poi_id_mean_delivery_distance', 400.51099829364375],
            ['lasthour_poi_agg_poi_id_std_arriveshop_cost', 400.48887587194884],
            ['cur_block_box_total_value_x', 400.46002985525098], ['15min', 400.44381665398993],
            ['lasthour_poi_agg_poi_id_mean_box_total_value', 400.43174890687999],
            ['lasthour_poi_agg_poi_id_std_delivery_distance', 400.43161778355011],
            ['cur_10min_not_fetched_order_num_y', 400.43096662061606],
            ['poi_agg_poi_id_arrive_guest_avg_speed_std', 400.41691612904606],
            ['cur_area_busy_coef_y', 400.4157042375486], ['cur_block_delivery_distance_x', 400.4007769678467],
            ['_10min_working_rider_num', 400.39327234264869],
            ['lasthour_poi_agg_poi_id_arrive_guest_avg_speed_mean', 400.38721452319135],
            ['poi_agg_poi_id_std_arrive_guest_cost', 400.38651928136386], ['poi_lng_bin', 400.37848462395795],
            ['cur_block_food_total_value_x', 400.3728383186924], ['cst_lng_bin', 400.36662120394857],
            ['box_total_value', 400.35090242289147],
            ['lasthour_poi_agg_poi_id_std_food_total_value', 400.34336202145818],
            ['lasthour_poi_agg_poi_id_mean_arrive_guest_cost', 400.33889647599767],
            ['poi_agg_poi_id_mean_box_total_value', 400.33700364559098],
            ['future_food_total_value_sum_x', 400.31835681270428],
            ['poi_agg_poi_id_mean_food_total_value', 400.31664439414101], ['direction', 400.31082583418691],
            ['poi_agg_poi_id_std_delivery_distance', 400.29234582354582],
            ['_10min_notbusy_working_rider_num', 400.2920281117357],
            ['cur_block_delivery_distance_y', 400.28780858389831],
            ['poi_agg_poi_id_arrive_guest_avg_speed_mean', 400.25435040342984],
            ['lasthour_poi_agg__avg_speed_stdpoi_id', 400.25059377560495],
            ['cur_waiting_order_num_y', 400.23034002417438], ['poi_lng', 400.21911597704383],
            ['area_id_area_id#hour#minute_arriveshop_cost_std', 400.21267307063937],
            ['cur_order_id_count', 400.2065364680771], ['cur_block_food_total_value', 400.2052012102381],
            ['lasthour_poi_agg_poi_id_std_fetch_cost', 400.19417569572562],
            ['_10min_not_fetched_order_num', 400.17663424908682], ['poi_agg_poi_id_coor_count', 400.17653417405501],
            ['food_num', 400.1720335813377], ['area_id_area_id#hour#minute_arriveshop_cost_mean', 400.16650134877187],
            ['poi_agg_poi_id_mean_dd', 400.16010654369893],
            ['poi_agg_poi_id_mean_arrive_guest_cost', 400.15019739443784],
            ['area_id_area_id#hour#minute_food_total_value_sum', 400.13130677746835],
            ['customer_longitude', 400.09409646842295],
            ['lasthour_poi_agg_poi_id_mean_waiting_order_num', 400.08948901124927],
            ['poi_agg_poi_id_mean_delivery_distance', 400.06016386646797],
            ['area_id_area_id#hour#minute_prd_arrive_guest_time_mean', 400.05659660464585],
            ['cur_block_food_total_value_y', 400.00587274364665], ['cur_waiting_order_num', 399.95815787596968],
            ['lasthour_poi_agg_poi_id_mean_fetch_cost', 399.87273351867083],
            ['poi_agg__avg_speed_stdpoi_id', 399.78779658574302], ['weekday', 399.67952325580671]]][::-1]


def init_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str, dest='train_path',
                        help="data")
    parser.add_argument('--test', type=str, dest='test_path',
                        help="data")
    parser.add_argument('--round', type=int, dest='round',
                        help="data", default=500)

    parser.add_argument('--output', type=str, dest='out_path',
                        help="data")
    return parser.parse_args()


if __name__ == '__main__':
    args = init_arguments()
    data = pd.read_csv(args.train_path)
    train_data = pd.read_csv(args.train_path)
    test_data = pd.read_csv(args.test_path)
    train_data.head(5)

    train_data.fillna(-999, inplace=True)
    test_data.fillna(-999, inplace=True)

    order_id = pd.DataFrame(np.unique(test_data.order_id), columns=["order_id"])

    labels = train_data[['delivery_duration']].values.astype(np.float32).flatten()
    labels_val = test_data[['delivery_duration']].values.astype(np.float32).flatten()

    rs = []
    score = 0
    best_drop = []
    for f in [None] + features:
        print "try without f {}".format(f)
        se_features = list(set(features) - set([f]) - set(best_drop))
        lgb_train = lgb.Dataset(train_data[se_features], labels)
        lgb_eval = lgb.Dataset(test_data[se_features], labels_val, reference=lgb_train)

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'mae'},
            'num_leaves': 256,
            'min_sum_hessian_in_leaf': 20,
            'max_depth': -12,
            'learning_rate': 0.05,
            'feature_fraction': 0.6,
            # 'bagging_fraction': 0.9,
            # 'bagging_freq': 3,
            'verbose': 1
        }

        print('Start training...')
        # train
        gbm = lgb.train(params,
                        lgb_train,
                        valid_sets=lgb_eval,
                        num_boost_round=args.round,
                        early_stopping_rounds=20,
                        verbose_eval=False
                        )

        # print('Feature names:', gbm.feature_name())

        # print('Calculate feature importances...')
        # feature importances
        # print('Feature importances:', list(gbm.feature_importance()))

        df = pd.DataFrame({'feature': gbm.feature_name(), 'importances': gbm.feature_importance()})
        # print(df.sort_values('importances'))

        test_data.loc[:, "delivery_duration_prd"] = gbm.predict(test_data[se_features])
        cur_s = mean_absolute_error(test_data["delivery_duration"], test_data["delivery_duration_prd"])
        print "cur f is {},last score is {}, cur_s is {}, ".format(f, score, cur_s)
        if score == 0 or score > cur_s:
            score = cur_s
            if f:
                print "best drop f {}".format(f)
                best_drop.append(f)

        rs.append([f if f else "all_features",
                   mean_absolute_error(test_data["delivery_duration"], test_data["delivery_duration_prd"])])
    print "final score is  {},best_drop is {}".format(score, best_drop)
    # rs = order_id.merge(test_data[["order_id", "delivery_duration"]], left_on="order_id", right_on="order_id",
    #                     how="left")
    #
    # rs.to_csv(args.out_path, header=['order_id', 'delivery_duration'], index=False)
    # print sorted(rs, key=lambda x: -x[1])
    # print "\n".join(map(lambda x: ":".join(map(str, x)), sorted(rs, key=lambda x: -x[1])))
