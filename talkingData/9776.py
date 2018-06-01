"""
A non-blending lightGBM model that incorporates portions and ideas from various public kernels.
"""
# DEBUG = True
import argparse


class tick_tock:
    def __init__(self, process_name, verbose=1):
        self.process_name = process_name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print(self.process_name + " begin ......")
            self.begin_time = time.time()

    def __exit__(self, type, value, traceback):
        if self.verbose:
            end_time = time.time()
            print(self.process_name + " end ......")
            print('time lapsing {0} s \n'.format(end_time - self.begin_time))


def init_arguments():
    def str_to_bool(s):
        if s == 't':
            return True
        elif s == 'f':
            return False
        else:
            raise ValueError

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', type=str_to_bool, dest='debug', default=True,
                        help="data")
    parser.add_argument('--val', type=str_to_bool, dest='val', default=False,
                        help="data")
    parser.add_argument('--nchunk', type=int, dest="nchunk", default=50000000,
                        help="map key(rider or rider_poi)")

    parser.add_argument('--offset', type=int, dest="offset", default=75000000,
                        help="map key(rider or rider_poi)")
    parser.add_argument('--dataroot', type=str, dest="data_root", default='/home/dongjian/data/talkingdata/',
                        help="map key(rider or rider_poi)")

    return parser.parse_args()


args = init_arguments()
DEBUG = args.debug
WHERE = 'kaggle'
FILENO = 7
NCHUNK = args.nchunk
OFFSET = args.offset  # 75000000
VAL_RUN = args.val

print(args)

MISSING32 = 999999999
MISSING8 = 255
PUBLIC_CUTOFF = 4032690

data_root = args.data_root
if WHERE == 'kaggle':
    inpath = data_root
    pickle_path = data_root
    suffix = '.zip'
    outpath = '/home/dongjian/work/talkingdata/sub/'
    savepath = ''
    oofpath = ''
    cores = 6
elif WHERE == 'gcloud':
    inpath = '../.kaggle/competitions/talkingdata-adtracking-fraud-detection/'
    pickle_path = '../data/'
    suffix = '.zip'
    outpath = '../sub/'
    oofpath = '../oof/'
    savepath = '../data/'
    cores = 7

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os


def do_count(df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True):
    if show_agg:
        print("Aggregating by ", group_cols, '...')
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return (df)


def do_countuniq(df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True):
    if show_agg:
        print("Counting unqiue ", counted, " by ", group_cols, '...')
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].nunique().reset_index().rename(
            columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return (df)


def do_cumcount(df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True):
    if show_agg:
        print("Cumulative count by ", group_cols, '...')
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name] = gp.values
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return (df)


def do_mean(df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True):
    if show_agg:
        print("Calculating mean of ", counted, " by ", group_cols, '...')
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].mean().reset_index().rename(
            columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return (df)


def do_var(df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True):
    if show_agg:
        print("Calculating variance of ", counted, " by ", group_cols, '...')
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return (df)


debug = DEBUG
if debug:
    print('*** debug parameter set: this is a test run for debugging purposes ***')

if VAL_RUN:
    nrows = 122071522
    outpath = oofpath
else:
    nrows = 184903890
nchunk = NCHUNK
val_size = 2500000
frm = nrows - OFFSET
if debug:
    frm = 0
    nchunk = 100000
    val_size = 10000
to = frm + nchunk
fileno = FILENO

dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32',
}

if VAL_RUN:
    print('loading train data...', frm, to)
    train_df = pd.read_pickle(pickle_path + "training.pkl.gz")[frm:to]
    train_df['click_time'] = pd.to_datetime(train_df.click_time)
    print('loading test data...')
    if debug:
        public_cutoff = 10000
        test_df = pd.read_pickle(pickle_path + "validation.pkl.gz")[:30000]
        test_df['click_time'] = pd.to_datetime(test_df.click_time)
        y_test = test_df['is_attributed'].values
        test_df.drop(['is_attributed'], axis=1, inplace=True)
    else:
        public_cutoff = PUBLIC_CUTOFF
        test_df = pd.read_pickle(pickle_path + "validation.pkl.gz")
        test_df['click_time'] = pd.to_datetime(test_df.click_time)
        y_test = test_df['is_attributed'].values
        test_df.drop(['is_attributed'], axis=1, inplace=True)
else:
    print('loading train data...', frm, to)
    train_df = pd.read_csv(inpath + "train.csv", parse_dates=['click_time'], skiprows=range(1, frm), nrows=to - frm,
                           dtype=dtypes,
                           usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])
    print(train_df.head(1))
    print(train_df.tail(1))
    print('loading test data...')
    if debug:
        test_df = pd.read_csv(inpath + "test.csv", nrows=100000, parse_dates=['click_time'], dtype=dtypes,
                              usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])
    else:
        test_df = pd.read_csv(inpath + "test.csv", parse_dates=['click_time'], dtype=dtypes,
                              usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])
    train_df['click_id'] = MISSING32
    train_df['click_id'] = train_df.click_id.astype('uint32')

len_train = len(train_df)
test_df['is_attributed'] = MISSING8
test_df['is_attributed'] = test_df.is_attributed.astype('uint8')
train_df = train_df.append(test_df)

del test_df
gc.collect()

print('Extracting new features...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')

print('Extracting aggregation features...')
train_df = do_cumcount(train_df, ['ip', 'device', 'os'], 'app', 'X1', show_max=True);
gc.collect()
train_df = do_cumcount(train_df, ['ip'], 'os', 'X7', show_max=True);
gc.collect()
train_df = do_countuniq(train_df, ['ip'], 'channel', 'X0', 'uint8', show_max=True);
gc.collect()
train_df = do_countuniq(train_df, ['ip', 'day'], 'hour', 'X2', 'uint8', show_max=True);
gc.collect()
train_df = do_countuniq(train_df, ['ip'], 'app', 'X3', 'uint8', show_max=True);
gc.collect()
train_df = do_countuniq(train_df, ['ip', 'app'], 'os', 'X4', 'uint8', show_max=True);
gc.collect()
train_df = do_countuniq(train_df, ['ip'], 'device', 'X5', 'uint16', show_max=True);
gc.collect()
train_df = do_countuniq(train_df, ['app'], 'channel', 'X6', show_max=True);
gc.collect()
train_df = do_countuniq(train_df, ['ip', 'device', 'os'], 'app', 'X8', show_max=True);
gc.collect()
train_df = do_count(train_df, ['ip', 'day', 'hour'], 'ip_tcount', show_max=True);
gc.collect()
train_df = do_count(train_df, ['ip', 'app'], 'ip_app_count', show_max=True);
gc.collect()
train_df = do_count(train_df, ['ip', 'app', 'os'], 'ip_app_os_count', 'uint16', show_max=True);
gc.collect()
train_df = do_var(train_df, ['ip', 'day', 'channel'], 'hour', 'ip_tchan_count', show_max=True);
gc.collect()
train_df = do_var(train_df, ['ip', 'app', 'os'], 'hour', 'ip_app_os_var', show_max=True);
gc.collect()
train_df = do_var(train_df, ['ip', 'app', 'channel'], 'day', 'ip_app_channel_var_day', show_max=True);
gc.collect()
train_df = do_mean(train_df, ['ip', 'app', 'channel'], 'hour', 'ip_app_channel_mean_hour', show_max=True);
gc.collect()

with tick_tock('Doing nextClick...') as f:
    predictors = []
    new_feature = 'nextClick'
    D = 2 ** 26
    train_df['category'] = (
                               train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df[
                                   'device'].astype(
                                       str) \
                               + "_" + train_df['os'].astype(str)).apply(hash) % D
    click_buffer = np.full(D, 3000000000, dtype=np.uint32)
    train_df['epochtime'] = train_df['click_time'].astype(np.int64) // 10 ** 9
    next_clicks = []
    for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
        next_clicks.append(click_buffer[category] - t)
        click_buffer[category] = t
    del (click_buffer)
    QQ = list(reversed(next_clicks))
    train_df[new_feature] = pd.Series(QQ).astype('float32')
    predictors.append(new_feature)
    del QQ, next_clicks
    gc.collect()

with tick_tock('diff feature ') as f:
    new_feature = 'times_in_future_diff_3_times_mean'
    f__times_in_future_diff_3_times_mean = 'times_in_future_diff_3_times_min'

    D = 2 ** 26
    click_buffer = np.full((2 ** 26, 5), np.array([-1] * 5), dtype=np.uint32)
    times_in_future_5min = []
    times_in_future_diff_3_times_min = []

    for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
        times_in_future_5min.append(np.mean(np.diff([x for x in click_buffer[category] if x > 0])))
        times_in_future_diff_3_times_min.append(np.std(np.diff([x for x in click_buffer[category] if x > 0])))

        click_buffer[category][0] = click_buffer[category][1]
        click_buffer[category][1] = click_buffer[category][2]
        click_buffer[category][2] = click_buffer[category][3]
        click_buffer[category][3] = click_buffer[category][4]

        click_buffer[category][4] = t

    del (click_buffer)

    train_df[new_feature] = pd.Series(list(reversed(times_in_future_5min))).astype('float32')
    train_df[f__times_in_future_diff_3_times_mean] = pd.Series(list(reversed(times_in_future_diff_3_times_min))).astype(
            'float32')

    predictors.append(new_feature)
    predictors.append(f__times_in_future_diff_3_times_mean)

    del times_in_future_5min, times_in_future_diff_3_times_min
    gc.collect()

with tick_tock('before feature ') as f:
    before_click = 'beforeClick'
    filename = 'beforeClick_%d_%d.csv' % (frm, to)
    # if os.path.exists(filename):
    #     print('loading from save file')
    #     before_click_list = pd.read_csv(filename).values
    # else:
    D = 2 ** 26
    click_buffer = np.full(D, 1000000000, dtype=np.uint32)

    # train_df['epochtime'] = train_df['click_time'].astype(np.int64) // 10 ** 9
    next_clicks = []
    for category, t in zip(train_df['category'].values, train_df['epochtime'].values):
        next_clicks.append(t - click_buffer[category])
        click_buffer[category] = t
    del (click_buffer)
    before_click_list = list(next_clicks)

    if not debug:
        print('saving')
        pd.DataFrame(before_click_list).to_csv(filename, index=False)

    train_df[before_click] = pd.Series(before_click_list).astype('float32')
    predictors += [before_click]

with tick_tock("doing past and reverse feature") as f:
    train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
    train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

    # train_df['id_past_last_3_mean'] = train_df.groupby(['ip', 'device', 'app', 'os'], as_index=False,
    #                                                    group_keys=False).epochtime.transform(
    #         lambda x: x.diff().iloc[:-3].mean()).fillna(-1).reset_index(drop=True).astype('uint32')
    # train_df['id_past_last_3_min'] = train_df.groupby(['ip', 'device', 'app', 'os'], as_index=False,
    #                                                   group_keys=False).epochtime.transform(
    #         lambda x: x.diff().iloc[:-3].min()).fillna(-1).reset_index(drop=True).astype('uint32')

    reverse_train_df = train_df[::-1]
    reverse_train_df = do_cumcount(reverse_train_df, ['ip', 'device', 'os', 'app'], 'channel', 'rX1', show_max=True);
    train_df['rX1'] = reverse_train_df['rX1'][::-1]
    #
    # reverse_train_df = do_cumcount(reverse_train_df, ['ip', 'app'], 'os', 'rX2', show_max=True);
    # train_df['rX2'] = reverse_train_df['rX2'][::-1]
    #
    # reverse_train_df = do_cumcount(reverse_train_df, ['ip', 'os'], 'app', 'rX3', show_max=True);
    # train_df['rX3'] = reverse_train_df['rX3'][::-1]
    #
    # reverse_train_df = do_cumcount(reverse_train_df, ['ip', 'device'], 'app', 'rX4', show_max=True);
    # train_df['rX4'] = reverse_train_df['rX4'][::-1]
    #
    reverse_train_df = do_cumcount(reverse_train_df, ['device', 'app'], 'os', 'rX5', show_max=True);
    train_df['rX5'] = reverse_train_df['rX5'][::-1]
    reverse_train_df = do_cumcount(reverse_train_df, ['channel', 'app'], 'os', 'rX6', show_max=True);
    train_df['rX6'] = reverse_train_df['rX6'][::-1]
    predictors += ["rX1", "rX5", "rX6"]
    #
    # reverse_train_df = do_cumcount(reverse_train_df, ['device'], 'app', 'rX6', show_max=True);
    # train_df['rX6'] = reverse_train_df['rX6'][::-1]
    #
    # reverse_train_df = do_cumcount(reverse_train_df, ['ip'], 'os', 'rX7', show_max=True);
    # train_df['rX7'] = reverse_train_df['rX7'][::-1]
    # predictors += ["rX" + str(x) for x in range(1, 7)]
    # ['id_past_last_3_mean', 'id_past_last_3_min'] + \

    # del reverse_train_df

print("vars and data type: ")
gc.collect()

train_df.drop(['epochtime', 'category', 'click_time'], axis=1, inplace=True)
train_df.info()

target = 'is_attributed'
predictors.extend(['app', 'device', 'os', 'channel', 'hour',
                   'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                   'ip_app_os_count', 'ip_app_os_var',
                   'ip_app_channel_var_day', 'ip_app_channel_mean_hour',
                   'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'])
categorical = ['app', 'device', 'os', 'channel', 'hour']
print('predictors', predictors)

test_df = train_df[len_train:]
val_df = train_df[(len_train - val_size):len_train]
train_df = train_df[:(len_train - val_size)]

print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))

test_df.to_pickle('test.pkl.gz')
del test_df
gc.collect()

print("Training...")
start_time = time.time()

objective = 'binary'
metrics = 'auc'
early_stopping_rounds = 30
verbose_eval = True
num_boost_round = 1000
categorical_features = categorical
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': objective,
    'metric': metrics,
    'learning_rate': 0.2,
    # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 7,  # 2^max_depth - 1
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 300,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight': 200,  # because training data is extremely unbalanced
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0,  # L2 regularization term on weights
    'nthread': cores,
    'verbose': 0,
    'metric': metrics
}

print("preparing validation datasets")
xgtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                      feature_name=predictors,
                      categorical_feature=categorical
                      )

print(train_df[predictors].head())
print(train_df[target].head())
print(val_df[predictors].head())
print(val_df[target].head())

del train_df
if WHERE != 'kaggle':
    xgtrain.save_binary('xgtrain.bin')
    del xgtrain
xgvalid = lgb.Dataset(val_df[predictors].values, label=val_df[target].values,
                      feature_name=predictors,
                      categorical_feature=categorical
                      )
del val_df
gc.collect()

evals_results = {}

if WHERE != 'kaggle':
    xgtrain = lgb.Dataset('xgtrain.bin',
                          feature_name=predictors,
                          categorical_feature=categorical
                          )

print(lgb_params)
bst = lgb.train(lgb_params,
                xgtrain,
                valid_sets=[xgtrain, xgvalid],
                valid_names=['train', 'valid'],
                evals_result=evals_results,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=10,
                feval=None)

print("\nModel Report")
print("bst.best_iteration: ", bst.best_iteration)
print(metrics + ":", evals_results['valid'][metrics][bst.best_iteration - 1])

print('[{}]: model training time'.format(time.time() - start_time))

if WHERE != 'gcloud':
    print('Plot feature importances...')
    df = pd.DataFrame({'feature': bst.feature_name(), 'importances': bst.feature_importance()})
    df['fscore'] = df['importances'] / df['importances'].sum()

    print(df.sort_values('importances', ascending=False).to_string(index=False))
    print("-" * 20 + " features" + "-" * 20)
    print([x.strip() for x in df.sort_values('importances', ascending=False).feature.to_string(index=False).split("\n")])

print("Re-reading test data...")
test_df = pd.read_pickle('test.pkl.gz')
sub = pd.DataFrame()

print("Predicting...")
y_pred = bst.predict(test_df[predictors], num_iteration=bst.best_iteration)
outsuf = ''
if VAL_RUN:
    print("\n\nFULL VALIDATION SCORE:    ",
          roc_auc_score(y_test, y_pred))
    print("PUBLIC VALIDATION SCORE:  ",
          roc_auc_score(y_test[:public_cutoff], y_pred[:public_cutoff]))
    print("PRIVATE VALIDATION SCORE: ",
          roc_auc_score(y_test[public_cutoff:], y_pred[public_cutoff:]))
    outsuf = '_val'
    sub['click_id'] = pd.Series(range(len(test_df))).astype('uint32').values
else:
    sub['click_id'] = test_df['click_id'].astype('uint32').values

if WHERE != 'kaggle':
    os.remove('xgtrain.bin')
os.remove('test.pkl.gz')
sub['is_attributed'] = y_pred
if not debug:
    print("\nwriting...")
    if WHERE == 'kaggle':
        sub.to_csv('sub_it%d' % (fileno) + outsuf + '.csv.gz', index=False, float_format='%.9f', compression='gzip')
    else:
        sub.to_csv('sub_it%d' % (fileno) + outsuf + '.csv.gz', index=False, float_format='%.9f', compression='gzip')
print("\ndone...")
print(sub.head(10))
