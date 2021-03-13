#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"


# %% [code]
# !ls ../input/lgmodels

# %% [code]
import warnings
warnings.filterwarnings("ignore")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob

from scipy.stats import describe
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [code]
LABELS = ["isFraud"]
all_files = glob.glob("../input/lgmodels/*.csv")
scores = np.zeros(len(all_files))
for i in range(len(all_files)):
    scores[i] = float('.'+all_files[i].split(".")[3])

# %% [code]
top = scores.argsort()[::-1]
for i, f in enumerate(top):
    print(i,scores[f],all_files[f])

# %% [code]
outs = [pd.read_csv(all_files[f], index_col=0) for f in top]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "m" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols

# %% [code]
# check correlation
corr = concat_sub.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(len(cols)+2, len(cols)+2))

# Draw the heatmap with the mask and correct aspect ratio
_ = sns.heatmap(corr,mask=mask,cmap='prism',center=0, linewidths=1,
                annot=True,fmt='.4f', cbar_kws={"shrink":.2})

# %% [code]
mean_corr = corr.mean()
mean_corr = mean_corr.sort_values(ascending=True)
mean_corr = mean_corr[:6]
mean_corr

# %% [code]
m_gmean1 = 0
for n in mean_corr.index:
    m_gmean1 += np.log(concat_sub[n])
m_gmean1 = np.exp(m_gmean1/len(mean_corr))

# %% [code]
rank = np.tril(corr.values,-1)
rank[rank<0.92] = 1
m = (rank>0).sum() - (rank>0.97).sum()
m_gmean2, s = 0, 0
for n in range(m):
    mx = np.unravel_index(rank.argmin(), rank.shape)
    w = (m-n)/m
    m_gmean2 += w*(np.log(concat_sub.iloc[:,mx[0]])+np.log(concat_sub.iloc[:,mx[1]]))/2
    s += w
    rank[mx] = 1
m_gmean2 = np.exp(m_gmean2/s)

# %% [code]
top_mean = 0
s = 0
for n in [0,1,3,7,26]:
    top_mean += concat_sub.iloc[:,n]*scores[top[n]]
    s += scores[top[n]]
top_mean /= s

# %% [code]
m_gmean = np.exp(0.3*np.log(m_gmean1) + 0.2*np.log(m_gmean2) + 0.5*np.log(top_mean))
describe(m_gmean)

# %% [code]
concat_sub['isFraud'] = m_gmean
concat_sub[['isFraud']].to_csv('stack_gmean.csv')







all_files2 = glob.glob("/tmp/lgmodels/*.csv")
all_files2.sort(key=lambda s: s.split('.')[1], reverse=True)

aa_outs = [pd.read_csv(all_files2[f], index_col=0) for f in range(len(all_files2))]
aa_concat_sub = pd.concat(aa_outs, axis=1)
# aa_concat_sub.columns = all_files

aa_corr = aa_concat_sub.corr()
mask = np.diag([True,]*len(aa_corr))
mm = np.ma.array(aa_corr, mask=mask)

for i in range(10):
    mi = np.unravel_index(mm.argmax(), mm.shape)
    print(f'max value at {mi} : {mm[mi]}')
    idx_to_mask = max(mi)
    mm.mask[idx_to_mask, :] = True
    mm.mask[:, idx_to_mask] = True

idx_to_drop = mm.mask.all(axis=0)

mmf = np.ma.array(all_files2, mask=idx_to_drop)

all_files = mmf.tolist()
all_files = list(filter(lambda x: x is not None, all_files))









