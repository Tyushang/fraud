#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"
import os

import numpy as np
import pandas as pd


DS_DIR = MINIFY_DIR = '../input/fraud-minify'

test_merge: pd.DataFrame = pd.read_pickle(os.path.join(DS_DIR, 'test_merge.pkl'))
train_merge: pd.DataFrame = pd.read_pickle(os.path.join(DS_DIR, 'train_merge.pkl'))


