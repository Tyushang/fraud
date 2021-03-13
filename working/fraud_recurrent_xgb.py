#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"

# _______________________________________________________________________________________
# imports:
import gc
import json
import warnings

import seaborn as sns

sns.set()

from fraud_utils import *
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')

# %matplotlib inline

# _______________________________________________________________________________________
# configs:
DS_DIR = eval('DS_DIR')
N_FOLD = 6

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

with open(os.path.join(DS_DIR, 'config.json'), 'r') as f:
    config = json.load(f)
    feat_columns = config['feat_columns']
    merge_cate_feat = config['merge_cate_feat']
    FILLNA_BY = config['fillna_by']

# _______________________________________________________________________________________
# load dataset:
print('Loading Data...')
train_merge: pd.DataFrame = pd.read_pickle(os.path.join(DS_DIR, 'train_merge.pkl'))
test_merge: pd.DataFrame = pd.read_pickle(os.path.join(DS_DIR, 'test_merge.pkl'))

for df in [train_merge, test_merge]:
    for col_name in df.select_dtypes('category').columns:
        df[col_name] = df[col_name].astype('float')
        df[col_name].fillna(FILLNA_BY, inplace=True)
        df[col_name] = df[col_name].astype('int')

train_merge.fillna(FILLNA_BY, inplace=True)
test_merge.fillna(FILLNA_BY, inplace=True)

folds = GroupKFold(n_splits=N_FOLD)

X_nontest = train_merge[feat_columns]
y_nontest = train_merge[TARGET]

X_test = test_merge[feat_columns]

boosters = []
oof = np.zeros(len(X_nontest))
predictions = np.zeros([len(X_test), N_FOLD])
for i_fold, (train_index, valid_index) in enumerate(folds.split(X_nontest, y_nontest, groups=train_merge['DT_M'])):
    xgbclf = xgb.XGBClassifier(
        n_estimators=5000,
        max_depth=12,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.4,
        missing=FILLNA_BY,
        tree_method='gpu_hist' if GPU_FOUND else 'hist',  # THE MAGICAL PARAMETER
        eval_metric='auc',
        # reg_alpha=0.15,
        # reg_lamdba=0.85,
        silent=False,
    )
    X_train, X_valid = X_nontest.iloc[train_index], X_nontest.iloc[valid_index]
    y_train, y_valid = y_nontest.iloc[train_index], y_nontest.iloc[valid_index]

    with timer(f'Fold-{i_fold}, xgbclf.fit: '):
        xgbclf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100, early_stopping_rounds=200)
    del X_train, y_train

    y_pred_valid = xgbclf.predict_proba(X_valid)[:, 1]
    print('ROC accuracy: {}'.format(roc_auc_score(y_valid, y_pred_valid)))
    oof[valid_index] = y_pred_valid
    del X_valid

    with timer(f'Fold-{i_fold}, xgbclf.predict_proba of test: '):
        predictions[:, i_fold] = xgbclf.predict_proba(X_test)[:, 1].flatten()
    boosters.append(xgbclf)

    gc.collect()

test_merge['isFraud'] = predictions.mean(axis=-1)
test_merge[['isFraud']].to_csv('sub_xgboost.csv', index=True)
