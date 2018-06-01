import numpy as np
import pandas as pd
import os

train_cv = "train_cv.csv"
test_cv = "test_cv.csv"

input_path = "/Users/dongjian/data/meituanKaggleData/"
import solutions.meituan_lasthour.cv_data_producer as cv
import solutions.meituan_lasthour.lgbm_cv as lc


def load_base_data(train_path,test_path):
	'''
	You can do your Feature engineering in this place
	x_train: x_data without lable
	x_test: you need to predict
	y_train: x_data label
	n_train: train column length
	n_test: test column length
	test_id
	'''
	train = pd.read_csv(train_path).fillna(0)
	test = pd.read_csv(test_path).fillna(0)
	features = list(set(train.columns.tolist()[1:]) - set(lc.to_drop) - set(lc.fea_drop))

	#--------------------------------------------------------main model---------------------------------------------------------------

	x_train = train[features]
	x_test = test[features]
	y_train = train["delivery_duration"].values.flatten()
	# y_train = train[["delivery_duration"]]

	return x_train, x_test, y_train, len(x_train), len(x_test)


