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
    parser.add_argument('--fname', type=str, dest='fea_name', default=25000000,
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


def do_prev_Click(df, agg_suffix='prevClick', agg_type='float32'):
    print(">> \nExtracting {0} time calculation features...\n".format(agg_suffix))

    GROUP_BY_NEXT_CLICKS = [

        # V1
        # {'groupby': ['ip']},
        {'groupby': ['ip', 'app']},
        # {'groupby': ['ip', 'channel']},
        {'groupby': ['ip', 'os']},

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

        # Run calculation

        # print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")

        # df[new_feature] = (df.click_time - df[all_features].groupby(spec[
        #                                                                 'groupby']).click_time.shift(
        #         +1)).dt.seconds.astype(agg_type)

        df['first_' + new_feature] = ((df.click_time - df[all_features].groupby(
                spec['groupby']).click_time.transform(
                'first')) / np.timedelta64(1, 's')).astype(agg_type)
        df['last_' + new_feature] = ((df[all_features].groupby(spec['groupby']).click_time.transform(
                'last') - df.click_time) / np.timedelta64(1, 's')).astype(agg_type)

        print new_feature, "new_feature"
        predictors.extend(['first_' + new_feature, 'last_' + new_feature])
        print
        gc.collect()
    return (df)


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
        train_df = pd.read_csv("/home/dongjian/data/talkingdata/train.csv", parse_dates=['click_time'],
                               skiprows=range(1, frm), nrows=to - frm,
                               dtype=dtypes,
                               usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])

    with tick_tock("loading test data...") as _:
        if debug:
            test_df = pd.read_csv("/home/dongjian/data/talkingdata/test.csv", nrows=100000, parse_dates=['click_time'],
                                  dtype=dtypes,
                                  usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])
        else:
            test_df = pd.read_csv("/home/dongjian/data/talkingdata/test.csv", parse_dates=['click_time'], dtype=dtypes,
                                  usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])
        if subsample:
            print "train df size bf subsample", len(train_df)
            print "train df size bf subsample", len(train_df)

            train_df = train_df[train_df.ip % 10 == 1]
            test_df = test_df[test_df.ip % 10 == 1]
            print "train df size af subsample", len(train_df)
            print "test df size af subsample", len(test_df)
    len_train = len(train_df)
    train_df = train_df.append(test_df)

    del test_df

    with tick_tock("next features"):
        train_df = do_prev_Click(train_df, agg_suffix='lagFea', agg_type='float32');
        gc.collect()
        gc.collect()

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
        print predictors
        print train_df.head()
        train_df[predictors].to_feather("train_{fname}_{frm}_{to}_{debug}_{subsample}.ftr".format(**
                                                                                                  {"debug": debug,
                                                                                                   "frm": frm,
                                                                                                   "to": to,
                                                                                                   "subsample": subsample,
                                                                                                   "fname": args.fea_name}))
        val_df[predictors].to_feather("val_{fname}_{frm}_{to}_{debug}_{subsample}.ftr".format(**
                                                                                              {"debug": debug,
                                                                                               "frm": frm, "to": to,
                                                                                               "subsample": subsample,
                                                                                               "fname": args.fea_name}))
        test_df[predictors].to_feather("test_{fname}_{frm}_{to}_{debug}_{subsample}.ftr".format(**
                                                                                                {"debug": debug,
                                                                                                 "frm": frm,
                                                                                                 "to": to,
                                                                                                 "subsample": subsample,
                                                                                                 "fname": args.fea_name}))


####### Chunk size defining and final run  ############

nrows = 184903891 - 1
nchunk = args.nchunk
val_size = args.val

frm = nrows - 65000000
if debug:
    frm = 0
    nchunk = 100000
    val_size = 10000

val_size = val_size / 10 if subsample else val_size

to = frm + nchunk

sub = DO(frm, to, FILENO)
