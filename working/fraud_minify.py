#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# General imports
import warnings

from fraud_utils import *

warnings.filterwarnings('ignore')


# _______________________________________________________________________________________
# configurations:
DS_DIR = eval('DS_DIR')

SEED = 42
seed_everything(SEED)
LOCAL_TEST = False


# _______________________________________________________________________________________
# load dataset
print('Load Data')
train_trans = pd.read_csv(os.path.join(DS_DIR, 'train_transaction.csv'), index_col='TransactionID')
test_trans = pd.read_csv(os.path.join(DS_DIR, 'test_transaction.csv'), index_col='TransactionID')
test_trans['isFraud'] = 0

train_id = pd.read_csv(os.path.join(DS_DIR, 'train_identity.csv'), index_col='TransactionID')
test_id = pd.read_csv(os.path.join(DS_DIR, 'test_identity.csv'), index_col='TransactionID')

# _______________________________________________________________________________________
# Final Minification
print('-'*30 + ' Reducing memory usage of train transaction...')
reduce_mem_usage(train_trans)
print('-'*30 + ' Reducing memory usage of test transaction...')
reduce_mem_usage(test_trans)

print('-'*30 + ' Reducing memory usage of train identity...')
reduce_mem_usage(train_id)
print('-'*30 + ' Reducing memory usage of test identity...')
reduce_mem_usage(test_id)

train_merge_encoded: pd.DataFrame = train_trans.merge(train_id, how='left', on='TransactionID')
test_merge_encoded: pd.DataFrame = test_trans.merge(test_id, how='left', on='TransactionID')

# _______________________________________________________________________________________
# Export
print('-'*30 + ' Pickling...')
train_trans.to_pickle('train_transaction.pkl')
test_trans.to_pickle('test_transaction.pkl')

train_id.to_pickle('train_identity.pkl')
test_id.to_pickle('test_identity.pkl')

train_merge_encoded.to_pickle('train_merge.pkl')
test_merge_encoded.to_pickle('test_merge.pkl')
