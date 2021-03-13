#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from tqdm import tqdm

import os

DS_DIR = '../input/ieee-fraud-detection/'

connect = create_engine(
    'mysql+pymysql://root:mysqlpass@localhost:3306/fraud?charset=utf8')


def csv_2_mysql(csv_path: str,
                con: Engine,
                table: str,
                index=True,
                index_label: str = None,
                if_exists='fail',
                chunksize: int = None):

    def to_mysql(df):
        pd.io.sql.to_sql(df, name=table, con=con,
                         index=index, index_label=index_label,
                         if_exists=if_exists)

    if chunksize is None:
        df = pd.read_csv(csv_path)
        to_mysql(df)
    else:
        chunks = pd.read_csv(csv_path, chunksize=chunksize)
        for chk in tqdm(chunks):
            to_mysql(chk)


def dump_wrapper(csv_name):
    print("Now dumping: " + csv_name)
    csv_2_mysql(os.path.join(DS_DIR, csv_name + '.csv'),
                table=csv_name,
                con=connect,
                index=False,
                index_label='TransactionID',
                chunksize=1000,
                if_exists='append')


dump_wrapper('train_identity')
dump_wrapper('train_transaction')

dump_wrapper('test_identity')
dump_wrapper('test_transaction')
dump_wrapper('sample_submission')

