#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"
import glob
import os

import numpy as np
import pandas as pd


SUB_DIR = '../input/submissions'
DS_DIR = '../input/ieee-fraud-detection'
TRAIN_TRANS_CSV = os.path.join(DS_DIR, 'train_transaction.csv')

# _______________________________________________________________________________________
# Remove highly relative submissions:
N_SUB_TO_KEEP = 10

N_SUB = len(os.listdir(SUB_DIR))
N_SUB_TO_DROP = N_SUB - N_SUB_TO_KEEP

all_file_path = glob.glob(os.path.join(SUB_DIR, '*.csv'))
all_file_path.sort(key=lambda s: s.split('.')[1], reverse=True)
all_file_path = all_file_path[:]

sub_df_lst = [pd.read_csv(all_file_path[f], index_col=0) for f in range(len(all_file_path))]
sub_df = pd.concat(sub_df_lst, axis=1)
# aa_concat_sub.columns = all_files

sub_corr = sub_df.corr()
mask = np.diag([True,] * len(sub_corr))
sub_corr_ma = np.ma.array(sub_corr, mask=mask)

for i in range(N_SUB_TO_DROP):
    i_max = np.unravel_index(sub_corr_ma.argmax(), sub_corr_ma.shape)
    print(f'max value at {i_max} : {sub_corr_ma[i_max]}')
    idx_to_mask = max(i_max)
    sub_corr_ma.mask[idx_to_mask, :] = True
    sub_corr_ma.mask[:, idx_to_mask] = True

idx_to_drop = sub_corr_ma.mask.all(axis=0)

all_file_path_with_mask = np.ma.array(all_file_path, mask=idx_to_drop)

file_path_to_keep = all_file_path_with_mask.tolist()
file_path_to_keep = list(filter(lambda x: x is not None, file_path_to_keep))
file_path_to_keep

# _______________________________________________________________________________________
# True target stats:
train_y = pd.read_csv(TRAIN_TRANS_CSV, usecols=['isFraud'])
E_T = train_y['isFraud'].mean()
D_T = E_T - E_T ** 2

# _______________________________________________________________________________________
# Pred target stats:
sub_files: str = file_path_to_keep
sub_concat: pd.DataFrame = pd.concat([pd.read_csv(f, index_col='TransactionID') for f in sub_files], axis=1)
sub_concat.columns = [f'sub-{sub_files.index(f)}' for f in sub_files]

sub_concat_mean: np.ndarray = sub_concat.mean(axis=0).values
P: np.ndarray = sub_concat.values - sub_concat_mean.reshape(1, -1)
Cov_P: np.ndarray = np.cov(P, rowvar=False)
# E_PiPj = Cov_P + E_Pi * E_Pj, and E_Pi = E_Pj = 0
A = E_PiPj = Cov_P
D_Pi = np.diag(Cov_P).reshape(-1, 1)

# _______________________________________________________________________________________
# True-Pred target stats:
auc = np.array([float('.' + f.split('.')[-2]) for f in sub_files])
corr_PiT = (2 * auc - 1).reshape(-1, 1)

B = E_PiT = np.sqrt(D_Pi * D_T) * corr_PiT

W = np.dot(np.linalg.inv(A), B)

S = np.dot(P, W)
S = (S - S.min()) / (S.max() - S.min())

Sub_blend = pd.DataFrame(S, index=sub_concat.index, columns=['isFraud'])

Sub_blend.to_csv('submission_blend.csv')

