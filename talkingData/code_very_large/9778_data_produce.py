"""
If you find this kernel helpful please upvote. Also any suggestion for improvement will be warmly welcomed.
I made cosmetic changes in the [code](https://www.kaggle.com/aharless/kaggle-runnable-version-of-baris-kanber-s-lightgbm/code).
Added some new features. Ran for 25mil chunk rows.
Also taken ideas from various public kernels.
"""
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
            print("#" * 50)
            print(self.process_name + " end ......")
            print(' time lapsing {0} s'.format(end_time - self.begin_time))
            print("#" * 50)


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
    parser.add_argument('--subsample', type=str_to_bool, dest='subsample', default=False,
                        help="data")
    parser.add_argument('--nchunk', type=int, dest='nchunk', default=25000000,
                        help="data")
    parser.add_argument('--frm', type=int, dest='frm', default=False,
                        help="data")
    parser.add_argument('--val', type=int, dest='val', default=25000000,
                        help="data")

    return parser.parse_args()


args = init_arguments()
FILENO = 5  # To distinguish the output file name.
debug = args.debug  # Whethere or not in debuging mode
subsample = args.subsample

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os

###### Feature extraction ######

#### Extracting next click feature
### Taken help from https://www.kaggle.com/nanomathias/feature-engineering-importance-testing
###Did some Cosmetic changes


predictors = []


def do_next_prev_Click(df, agg_suffix, agg_type='float32'):
    print('Extracting new features...')
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('int8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('int8')

    #### New added
    df['minute'] = pd.to_datetime(df.click_time).dt.minute.astype('int8')
    predictors.append('minute')
    df['second'] = pd.to_datetime(df.click_time).dt.second.astype('int8')
    predictors.append('second')
    print(">> \nExtracting {0} time calculation features...\n".format(agg_suffix))

    GROUP_BY_NEXT_CLICKS = [

        # V1
        # {'groupby': ['ip']},
        # {'groupby': ['ip', 'app']},
        # {'groupby': ['ip', 'channel']},
        # {'groupby': ['ip', 'os']},

        # V3
        {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
        {'groupby': ['ip', 'os', 'device']},
        {'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:

        # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']), agg_suffix)

        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']
        print(all_features)
        # Run calculation
        print(">> Grouping by {spec}, and saving time to {agg_suffix} in: {new_feature}".format(
                **{"agg_suffix": agg_suffix, "new_feature": new_feature, "spec": spec['groupby']}))

        if agg_suffix == "nextClick":
            df[new_feature] = (df[all_features].groupby(spec[
                                                            'groupby']).click_time.shift(
                    -1) - df.click_time).dt.seconds.astype(agg_type)
        # elif agg_suffix == "prevClick":
        #     df[new_feature] = (df.click_time - df[all_features].groupby(spec[
        #                                                                     'groupby']).click_time.shift(
        #             +1)).dt.seconds.astype(agg_type)
        predictors.append(new_feature)
        gc.collect()
    # print('predictors',predictors)
    return (df)


def do_prev_Click(df, agg_suffix='prevClick', agg_type='float32'):
    print(">> \nExtracting {0} time calculation features...\n".format(agg_suffix))

    GROUP_BY_NEXT_CLICKS = [

        # V1
        # {'groupby': ['ip']},
        # {'groupby': ['ip', 'app']},
        {'groupby': ['ip', 'channel']},
        {'groupby': ['ip', 'os']},

        # V3
        # {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
        # {'groupby': ['ip', 'os', 'device']},
        # {'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
        # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']), agg_suffix)

        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation

        # print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        df[new_feature] = (df.click_time - df[all_features].groupby(spec[
                                                                        'groupby']).click_time.shift(
                +1)).dt.seconds.astype(agg_type)

        predictors.append(new_feature)
        gc.collect()
    return (df)


## Below a function is written to extract count feature by aggregating different cols
def do_count(df, group_cols, agg_type='uint32', show_max=False, show_agg=True):
    agg_name = '{}count'.format('_'.join(group_cols))
    if show_agg:
        print("\nAggregating by ", group_cols, '... and saved in', agg_name)
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    #     print('predictors',predictors)
    gc.collect()
    return (df)


##  Below a function is written to extract unique count feature from different cols
def do_countuniq(df, group_cols, counted, agg_type='uint32', show_max=False, show_agg=True):
    agg_name = '{}_by_{}_countuniq'.format(('_'.join(group_cols)), (counted))
    if show_agg:
        print("\nCounting unqiue ", counted, " by ", group_cols, '... and saved in', agg_name)
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].nunique().reset_index().rename(
            columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    #     print('predictors',predictors)
    gc.collect()
    return (df)


### Below a function is written to extract cumulative count feature  from different cols
def do_cumcount(df, group_cols, counted, agg_type='uint32', show_max=False, show_agg=True):
    agg_name = '{}_by_{}_cumcount'.format(('_'.join(group_cols)), (counted))
    if show_agg:
        print("\nCumulative count by ", group_cols, '... and saved in', agg_name)
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name] = gp.values
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    #     print('predictors',predictors)
    gc.collect()
    return (df)


### Below a function is written to extract mean feature  from different cols
def do_mean(df, group_cols, counted, agg_type='float32', show_max=False, show_agg=True):
    agg_name = '{}_by_{}_mean'.format(('_'.join(group_cols)), (counted))
    if show_agg:
        print("\nCalculating mean of ", counted, " by ", group_cols, '... and saved in', agg_name)
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].mean().reset_index().rename(
            columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    #     print('predictors',predictors)
    gc.collect()
    return (df)


def do_var(df, group_cols, counted, agg_type='float32', show_max=False, show_agg=True):
    agg_name = '{}_by_{}_var'.format(('_'.join(group_cols)), (counted))
    if show_agg:
        print("\nCalculating variance of ", counted, " by ", group_cols, '... and saved in', agg_name)
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    #     print('predictors',predictors)
    gc.collect()
    return (df)


###  A function is written to train the lightGBM model with different given parameters
if debug:
    print('*** debug parameter set: this is a test run for debugging purposes ***')


## Running the full calculation.

#### A function is written here to run the full calculation with defined parameters.

def DO(frm, to, fileno):
    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint8',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'uint32',
    }

    with tick_tock('loading train data...{0} {1}'.format(frm, to)) as f:
        train_df = pd.read_csv("train.csv", parse_dates=['click_time'],
                               skiprows=range(1, frm), nrows=to - frm,
                               dtype=dtypes,
                               usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])

    with tick_tock("loading test data...") as _:
        if debug:
            test_df = pd.read_csv("test.csv", nrows=100000, parse_dates=['click_time'],
                                  dtype=dtypes,
                                  usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])
        else:
            test_df = pd.read_csv("test.csv", parse_dates=['click_time'], dtype=dtypes,
                                  usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])
        if subsample:
            # print("train df size bf subsample", len(train_df)
            print("train df size bf subsample", len(train_df))

            train_df = train_df[train_df.ip % 5 == 1]
            # test_df = test_df[test_df.ip % 10 == 1]
            print("train df size af subsample", len(train_df))
            # print("test df size af subsample", len(test_df)
    len_train = len(train_df)
    train_df = train_df.append(test_df)

    del test_df

    with tick_tock("next features"):
        train_df = do_next_prev_Click(train_df, agg_suffix='prevClick', agg_type='float32');
        # gc.collect()
        # train_df = do_prev_Click(train_df, agg_suffix='prevClick', agg_type='float32');
        # gc.collect()
        gc.collect()

    with tick_tock("mean features"):
        # train_df = do_next_prev_Click( train_df,agg_suffix='prevClick', agg_type='float32'  ); gc.collect()  ## Removed temporarily due RAM sortage.
        train_df = do_countuniq(train_df, ['ip'], 'channel');
        gc.collect()
        train_df = do_countuniq(train_df, ['ip', 'device', 'os'], 'app');
        gc.collect()
        train_df = do_countuniq(train_df, ['ip', 'day'], 'hour');
        gc.collect()
        train_df = do_countuniq(train_df, ['ip'], 'app');
        gc.collect()
        train_df = do_countuniq(train_df, ['ip', 'app'], 'os');
        gc.collect()
        train_df = do_countuniq(train_df, ['ip'], 'device');
        gc.collect()
        train_df = do_countuniq(train_df, ['app'], 'channel');
        gc.collect()
        train_df = do_cumcount(train_df, ['ip'], 'os');
        gc.collect()
        train_df = do_cumcount(train_df, ['ip', 'device', 'os'], 'app');
        gc.collect()
        train_df = do_count(train_df, ['ip', 'day', 'hour']);
        gc.collect()
        train_df = do_count(train_df, ['ip', 'app']);
        gc.collect()
        train_df = do_count(train_df, ['ip', 'app', 'os']);
        gc.collect()
        # train_df = do_var(train_df, ['ip', 'day', 'channel'], 'hour');
        gc.collect()
        train_df = do_var(train_df, ['ip', 'app', 'os'], 'hour');
        gc.collect()
        # train_df = do_var(train_df, ['ip', 'app', 'channel'], 'day');
        gc.collect()
        # train_df = do_mean(train_df, ['ip', 'app', 'channel'], 'hour');
        gc.collect()

        print(train_df.head(5))
        gc.collect()

    print('\n\nBefore appending predictors...\n\n', sorted(predictors))
    target = 'is_attributed'
    word = ['app', 'device', 'os', 'channel', 'hour', 'day', 'minute', 'second']
    for feature in word:
        if feature not in predictors:
            predictors.append(feature)
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day', 'minute', 'second']
    print('\n\nAfter appending predictors...\n\n', sorted(predictors))
    print(",".join(predictors))
    test_df = train_df[len_train:].reset_index()
    val_df = train_df[(len_train - val_size):len_train].reset_index()
    train_df = train_df[:(len_train - val_size)].reset_index()

    print("\ntrain size: ", len(train_df))
    print("\nvalid size: ", len(val_df))
    print("\ntest size : ", len(test_df))

    # train_df.to_csv("train_{frm}_{to}_{debug}_{subsample}.csv.gz".format(**
    #                                                                      {"debug": debug, "frm": frm, "to": to,
    #                                                                       "subsample": subsample}), index=False,
    #                 float_format='%.9f',
    #                 compression='gzip')
    # val_df.to_csv("val_{frm}_{to}_{debug}_{subsample}.csv.gz".format(**
    #                                                                  {"debug": debug, "frm": frm, "to": to,
    #                                                                   "subsample": subsample}), index=False,
    #               float_format='%.9f',
    #               compression='gzip')
    # test_df.to_csv("test_{frm}_{to}_{debug}_{subsample}.csv.gz".format(**
    #                                                                    {"debug": debug, "frm": frm, "to": to,
    #                                                                     "subsample": subsample}), index=False,
    #                float_format='%.9f',
    #                compression='gzip')

    with tick_tock("store data"):
        train_df.to_feather("train_{frm}_{to}_{debug}_{subsample}.ftr".format(**
                                                                              {"debug": debug, "frm": frm, "to": to,
                                                                               "subsample": subsample}))
        val_df.to_feather("val_{frm}_{to}_{debug}_{subsample}.ftr".format(**
                                                                          {"debug": debug, "frm": frm, "to": to,
                                                                           "subsample": subsample}))
        test_df.to_feather("test_{frm}_{to}_{debug}_{subsample}.ftr".format(**
                                                                            {"debug": debug, "frm": frm, "to": to,
                                                                             "subsample": subsample}))


####### Chunk size defining and final run  ############

nrows = 184903891 - 1
nchunk = args.nchunk
val_size = args.val

frm = 1
# frm = nrows - args.frm

if debug:
    frm = 0
    nchunk = 100000
    val_size = 10000

val_size = val_size / 5 if subsample else val_size

to = frm + nchunk

sub = DO(frm, to, FILENO)
