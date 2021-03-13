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
from sklearn.model_selection import KFold


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

for df in [train_merge, test_merge]:
    for col_name in df.select_dtypes('category').columns:
        df[col_name] = df[col_name].astype('float')
        df[col_name].fillna(-999, inplace=True)
        df[col_name] = df[col_name].astype('int')

train_merge.fillna(-999, inplace=True)
test_merge.fillna(-999, inplace=True)

with open(os.path.join(DS_DIR, 'merge_cate_feat.json'), 'r') as f:
    merge_cate_feat = json.load(f)


folds = KFold(n_splits=N_FOLD,shuffle=True)

feat_columns = [x for x in train_merge.columns if x not in ['TransactionID', 'TransactionDT', TARGET]]

X_nontest = train_merge[feat_columns]
y_nontest = train_merge[TARGET]

X_test = test_merge[feat_columns]

predictions = np.zeros([len(X_test), N_FOLD])
for i_fold, (train_index, valid_index) in enumerate(folds.split(X_nontest, y_nontest)):
    xgbclf = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=9,
        learning_rate=0.048,
        subsample=0.85,
        colsample_bytree=0.85,
        missing=-999,
        tree_method='gpu_hist' if GPU_FOUND else 'hist',  # THE MAGICAL PARAMETER
        reg_alpha=0.15,
        reg_lamdba=0.85,
        silent=False,
    )
    X_train, X_valid = X_nontest.iloc[train_index], X_nontest.iloc[valid_index]
    y_train, y_valid = y_nontest.iloc[train_index], y_nontest.iloc[valid_index]

    with timer(f'Fold-{i_fold}, xgbclf.fit: '):
        xgbclf.fit(X_train, y_train, )  # eval_set=(X_valid, y_valid), early_stopping_rounds=100)
    del X_train, y_train

    y_pred_valid = xgbclf.predict_proba(X_valid)[:, 1]
    del X_valid

    print('ROC accuracy: {}'.format(roc_auc_score(y_valid, y_pred_valid)))

    with timer(f'Fold-{i_fold}, xgbclf.predict_proba of test: '):
        predictions[:, i_fold] = xgbclf.predict_proba(X_test)[:, 1].flatten()

    gc.collect()

test_merge['isFraud'] = predictions.mean(axis=-1)
test_merge[['isFraud']].to_csv('sub_xgboost.csv', index=True)
