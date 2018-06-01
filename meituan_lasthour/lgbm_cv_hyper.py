from __future__ import print_function
import codecs
import math
import random
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
# import cleanup
import time
import colorama  # https://pypi.python.org/pypi/colorama
import pandas as pd
import numpy as np
import os
import platform
import itertools
from sklearn.model_selection import train_test_split
import lightgbm
from sklearn.preprocessing import MinMaxScaler
from uuid import uuid4
space ={
        #'boosting_type': hp.choice( 'boosting_type', ['gbdt', 'dart' ] ),
        #'max_depth': hp.quniform("max_depth", 4, 6, 1),
        'num_leaves': hp.quniform ('num_leaves', 20, 100, 1),
        'min_data_in_leaf':  hp.quniform ('min_data_in_leaf', 10, 100, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.75, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.75, 1.0),
        'learning_rate': hp.loguniform('learning_rate', -6.9, -2.3),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
        #'lambda_l1': hp.uniform('lambda_l1', 1e-4, 1e-6 ),
        #'lambda_l2': hp.uniform('lambda_l2', 1e-4, 1e-6 ),
        'seed': hp.randint('seed',2000000)
       }

best = hyperopt.fmin(fn=objective,
                     space=space,
                     algo=HYPEROPT_ALGO,
                     max_evals=N_HYPEROPT_PROBES,
                     trials=trials,
                     verbose=1)

print('-'*50)
print('The best params:')
print( best )
print('\n\n')