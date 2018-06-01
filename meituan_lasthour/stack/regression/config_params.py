import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV


def lgb_paras_regressor(leaves=7, method="regression", round=1000, sum_hessian=10, learing_rate=0.01, feature_fraction=0.8,
                        ):
    return {
        'task': 'train',
        'boosting_type': 'gbdt',
        "num_boost_round": round,
        'objective': method,
        'metric': {'mae'},
        "min_data_in_leaf":10,
        'num_leaves': 2 ** leaves,
        'min_sum_hessian_in_leaf': sum_hessian,
        'learning_rate': learing_rate,
        'feature_fraction': feature_fraction,
        'verbose': 1
    }



lgb_params_regressor_5 = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'mae'},
    'num_leaves': 2 ** 8,
    'min_sum_hessian_in_leaf': 10,
    'max_depth': -12,
    'learning_rate': 0.05,
    'feature_fraction': 0.6,
    'verbose': 1
}

lgb_params_regressor_8 = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'mae'},
    'num_leaves': 2 ** 10,
    'min_sum_hessian_in_leaf': 10,
    'max_depth': -12,
    'learning_rate': 0.05,
    'feature_fraction': 0.6,
    'verbose': 1
}

xgb_params_level_5 = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    # 'eval_metric': 'rsme',
    'eta': 0.15,
    # 'num_round': 1000,
    'colsample_bytree': 0.65,
    'subsample': 0.8,
    'max_depth': 5,
    'n_jobs': -1,
    'seed': 20171001,
    'silent': 1,
}

xgb_params_level_8 = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    # 'eval_metric': 'rsme',
    'eta': 0.15,
    # 'num_round': 1000,
    'colsample_bytree': 0.65,
    'subsample': 0.8,
    'max_depth': 9,
    'n_jobs': -1,
    'seed': 20171001,
    'silent': 1,
}

xgb_params_level_10 = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    # 'eval_metric': 'rsme',
    'eta': 0.15,
    # 'num_round': 1000,
    'colsample_bytree': 0.65,
    'subsample': 0.8,
    'max_depth': 11,
    'n_jobs': -1,
    'seed': 20171001,
    'silent': 1,
}


def build_xgb(n_iter, cv, eval_set):
    """
    Build a RandomSearchCV XGBoost model
    Parameters
    ----------
    n_iter : int
        Number of hyperparameters to try for RandomSearchCV
    cv : int
        Number of cross validation for RandomSearchCV
    eval_set : list of tuple
        List of (X, y) pairs to use as a validation set for
        XGBoost model's early-stopping
    Returns
    -------
    xgb_tuned : sklearn's RandomSearchCV object
        Unfitted RandomSearchCV XGBoost model
    """

    # for xgboost, set number of estimator to a large number
    # and the learning rate to be a small number, we'll simply
    # let early stopping decide when to stop training;
    xgb_params_fixed = {
        # setting it to a positive value
        # might help when class is extremely imbalanced
        # as it makes the update more conservative

        'n_jobs': -1}
    xgb = XGBClassifier(**xgb_params_fixed)

    # set up randomsearch hyperparameters:
    # subsample, colsample_bytree and max_depth are presumably the most
    # common way to control under/overfitting for tree-based models
    xgb_tuned_params = {
        'max_depth': randint(low=3, high=12),
        'colsample_bytree': uniform(loc=0.8, scale=0.2),
        'subsample': uniform(loc=0.8, scale=0.2),
        'max_delta_step': 1,
        'learning_rate': 0.1,
        'n_estimators': 500,}

    xgb_fit_params = {
        'eval_metric': 'auc',
        'eval_set': eval_set,
        'early_stopping_rounds': 5,
        'verbose': False}

    # computing the scores on the training set can be computationally
    # expensive and is not strictly required to select the parameters
    # that yield the best generalization performance.
    xgb_tuned = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=xgb_tuned_params,
            fit_params=xgb_fit_params,
            cv=cv,
            n_iter=n_iter,
            n_jobs=-1,
            verbose=1,
            return_train_score=False)
    return xgb_tuned

# def get_config_params(level_step):
#     if level_step == 1:
#         return xgb_params_level1
#     if level_step == 2:
#         return lgb_params_regressor
#     if level_step == 3:
#         return xgb_params_level2
