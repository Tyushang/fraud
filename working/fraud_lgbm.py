#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"

# _______________________________________________________________________________________
# imports:
import gc
import json
import warnings
from datetime import datetime

import lightgbm as lgb
from lightgbm.basic import Booster
from sklearn.model_selection import KFold

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

from fraud_utils import *

warnings.filterwarnings('ignore')

# %matplotlib inline

# _______________________________________________________________________________________
# configs:
DS_DIR = eval('DS_DIR')
N_FOLD = 3

SEED = 42
# seed_everything(SEED)
LOCAL_TEST = False
TARGET = 'isFraud'

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_FOUND = True
except:
    GPU_FOUND = False

# _______________________________________________________________________________________
# load dataset:
print('Loading Data...')
train_merge: pd.DataFrame = pd.read_pickle(os.path.join(DS_DIR, 'train_merge.pkl'))
test_merge: pd.DataFrame = pd.read_pickle(os.path.join(DS_DIR, 'test_merge.pkl'))

with open(os.path.join(DS_DIR, 'merge_cate_feat.json'), 'r') as f:
    merge_cate_feat = json.load(f)


# _______________________________________________________________________________________
# functions:
def lgb_train(params,
              df_to_train: pd.DataFrame,
              feature_columns: List[str],
              target: str,
              n_fold=5,
              verbose_eval=10,
              save_path=None) -> List[Booster]:
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=SEED)

    X, y = df_to_train[feature_columns], df_to_train[target]

    bs = []

    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        print('='*50 + f'Fold: {i_fold}')
        x_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        x_valid, y_valid = X.iloc[valid_idx, :], y.iloc[valid_idx]

        print(len(x_train), len(x_valid))
        train_ds = lgb.Dataset(x_train, label=y_train)
        valid_ds = lgb.Dataset(x_valid, label=y_valid)

        booster: Booster = lgb.train(
            params,
            train_ds,
            valid_sets=[train_ds, valid_ds],
            verbose_eval=verbose_eval,
        )
        bs.append(booster)
        if save_path is not None:
            booster.save_model(os.path.join(save_path, f'lgb-{i_fold}.txt'))

        del x_train, y_train, x_valid, y_valid, train_ds, valid_ds
        gc.collect()

    return bs


def lgb_predict(bs: List[Booster], df_to_predict, feature_columns: List[str], model_path: List[str]=None):
    effect_df = df_to_predict[feature_columns]
    n_boosters = len(bs)

    predictions = np.zeros([len(df_to_predict), n_boosters])
    for i_booster, booster in enumerate(bs):
        predictions[:, i_booster] = booster.predict(effect_df)

    return predictions


# _______________________________________________________________________________________
# Model params:
lgb_params = {
    # Static Variables
    'objective': 'binary',
    #     'num_class': [3],
    'metric': 'auc',
    'categorical_feature': merge_cate_feat,
    'learning_rate': 0.05, # Multiplication performed on each boosting iteration.
    'device': 'gpu' if GPU_FOUND else 'cpu', # GPU usage.
    'tree_learner': 'serial',
    'boost_from_average': 'true',
    'num_boost_round': 1000,
    'save_binary': True,

    #     # Dynamic Variables
    #     # https://sites.google.com/view/lauraepp/parameters
    'boosting_type': 'gbdt',#, 'goss', 'dart'],

    # Bushi-ness Parameters
    'max_depth': -1,  # -1 means no tree depth limit
    'num_leaves': 399, # we should let it be smaller than 2^(max_depth)

    # Tree Depth Regularization
    'subsample_for_bin': 3138, # Number of samples for constructing bin
    'min_data_in_leaf': 41, # Minimum number of data need in a child(min_data_in_leaf) - Must be motified when using a smaller dataset
    #     'min_gain_to_split': [0], # Prune by minimum loss requirement.
    'min_sum_hessian_in_leaf': 6.0609167205529655, # Prune by minimum hessian requirement - Minimum sum of instance weight(hessian) needed in a child(leaf)

    # Regularization L1/L2
    'reg_alpha': 2.30944428346164, # L1 regularization term on weights (0 is no regular)
    'reg_lambda': 2.7725874427554342, # L2 regularization term on weights
    'max_bin': 63,  # Number of bucketed bin for feature values

    # Row/Column Sampling
    #     'colsample_bytree': list(np.linspace(0.2, 1, 10).round(2)), # Subsample ratio of columns when constructing each tree.
    #         'subsample': parameters[6], # Subsample ratio of the training instance.
    #     'subsample_freq': 0, # frequence of subsample, <=0 means no enable
    'bagging_fraction': 0.9491459617974198,# Percentage of rows used per iteration frequency.
    'bagging_freq': 1,# Iteration frequency to update the selected rows.
    'feature_fraction': 0.6040170224661715, # Percentage of columns used per iteration.
    #     'colsample_bylevel': [1], # DANGER - Note Recommended Tuning - Percentage of columns used per split selection.

    # Dart Specific
    #     'max_drop': list(np.linspace(1, 70, 5).round(0).astype(int)), # Maximum number of dropped trees on one iteration.
    #     'rate_drop': list(np.linspace(0, .8, 10).round(2)), # Dropout - Probability to to drop a tree on one iteration.
    #     'skip_drop': list(np.linspace(.4, .6, 3).round(2)), # Probability of skipping any drop on one iteration.
    #     'uniform_drop': [False], # Uniform weight application for trees.

    # GOSS Specific
    #     'top_rate': [.2], # Keep top gradients.
    #     'other_rate': [.1], # Keep bottom gradients.
    # When top_rate + other_rate <= 0.5, the first iteration is sampled by (top_rate + other_rate)%. Attempts to keep only the bottom other_rate% gradients per iteration.

    # Imbalanced Dependent Variable
    'is_unbalance': False, #True if int(parameters[8]) == 1 else False, # because training data is unbalance (replaced with scale_pos_weight)
    #     'scale_pos_weight': []
    'nthread': -1, # Multi-threading
    'verbose': -1, # Logging Iteration Progression
    'seed': SEED # Seed for row sampling RNG.
}
feat_columns = [x for x in train_merge.columns if x not in ['TransactionID', 'TransactionDT', TARGET]]

# _______________________________________________________________________________________
# train / predict / submit.
with timer('lgb_train'):
    boosters = lgb_train(lgb_params,
                         train_merge,
                         feat_columns,
                         TARGET,
                         n_fold=N_FOLD,
                         save_path='.',
                         verbose_eval=20)

with timer('lgb_predict'):
    predicts = lgb_predict(boosters,
                           df_to_predict=test_merge,
                           feature_columns=feat_columns)

test_merge['isFraud'] = predicts.mean(axis=-1)
test_merge[['isFraud']].to_csv('submission.csv', index=True)

pd.DataFrame(predicts).to_csv('predicts.csv')

# _______________________________________________________________________________________
# plot importances.
imps: pd.DataFrame = pd.concat([pd.Series(b.feature_importance(), index=b.feature_name()) for b in boosters], axis=1)

avg_imp = pd.DataFrame({'average': imps.mean(axis=1)}).reset_index().rename(columns={'index': 'feature'})

plt.figure(figsize=(16, 16))
sns.barplot(data=avg_imp.sort_values(by='average', ascending=False).head(50), x='average', y='feature')
plt.title(f'50 TOP feature importance over {len(boosters)} folds average')
