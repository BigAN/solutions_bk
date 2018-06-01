import pandas as pd
import numpy as np
import xgboost as xgb
import datetime


input_path = '/Users/dongjian/data/meituanKaggleData/'

def load_order_data(file_name):
    df = pd.read_csv('%s/%s' % (input_path, file_name))
    c = 'order_unix_time'
    mask = pd.notnull(df[c])
    df.loc[mask, c] = df.loc[mask, c].apply(lambda x: datetime.datetime.fromtimestamp(x))
    df.loc[mask, 'date'] = df.loc[mask, c].apply(lambda x: x.strftime('%Y%m%d'))
    df.loc[mask, 'hour'] = df.loc[mask, c].apply(lambda x: x.hour)
    df.loc[mask, 'minute'] = df.loc[mask, c].apply(lambda x: x.minute)
    return df

def load_area_data(file_name):
    df = pd.read_csv('%s/%s' % (input_path, file_name), dtype={'date': str, 'time': str})
    mask = pd.notnull(df['time'])
    df.loc[mask, 'hour'] = df.loc[mask, 'time'].apply(lambda x: int(x[:2]))
    df.loc[mask, 'minute'] = df.loc[mask, 'time'].apply(lambda x: int(x[2:]))
    df.drop(['log_unix_time', 'time'], axis=1, inplace=True)
    return df

print('loading data...')
df_tr = load_order_data('waybill_info.csv')
mask = (df_tr.delivery_duration < 4654.0) & (df_tr.delivery_duration > 663.0) & ((df_tr.hour.values == 11) | (df_tr.hour.values == 17))
df_tr = df_tr.loc[mask]
df_te = load_order_data('waybill_info_test_b.csv')

df_tr_weather = load_area_data('weather_realtime.csv')
df_te_weather = load_area_data('weather_realtime_test.csv')

df_tr_area = load_area_data('area_realtime.csv')
df_te_area = load_area_data('area_realtime_test.csv')

print('merging data...')
df_tr = pd.merge(df_tr, df_tr_weather, on=['date', 'hour', 'minute', 'area_id'], how='left')
df_tr = pd.merge(df_tr, df_tr_area, on=['date', 'hour', 'minute', 'area_id'], how='left')

df_te = pd.merge(df_te, df_te_weather, on=['date', 'hour', 'minute', 'area_id'], how='left')
df_te = pd.merge(df_te, df_te_area, on=['date', 'hour', 'minute', 'area_id'], how='left')

print('constructing training data...')
cols = df_tr.columns.tolist()
to_drop = ['order_unix_time', 'arriveshop_unix_time', 'fetch_unix_time', 'finish_unix_time', 'order_id', 'delivery_duration', 'date']
features = list(np.setdiff1d(cols, to_drop))
print(features)

x_train = df_tr[features]
y_train = df_tr['delivery_duration']

x_test = df_te[features]
id_test = df_te['order_id']

print(x_train.shape)
print(x_test.shape)

dtrain = xgb.DMatrix(x_train.values, y_train)
dtest = xgb.DMatrix(x_test.values)

print('training model...')
watchlist = [(dtrain, 'train')]
param = {
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'eta': 0.15,
        'num_round': 1000,
        'colsample_bytree': 0.65,
        'subsample': 0.8,
        'max_depth': 5,
        'nthread': -1,
        'seed': 20171001,
        'silent': 1,
    }
bst = xgb.train(param, dtrain, param['num_round'], watchlist, verbose_eval=10)

print('generating prediction...')
pred = bst.predict(dtest)

print('generating submission...')
sub = pd.DataFrame({'order_id': id_test, 'delivery_duration': pred})

print('saving submission...')
sub.to_csv('sub_xgb_starter.csv', index=False)