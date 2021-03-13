#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"


import os
from multiprocessing.pool import Pool

import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm

DS_DIR = '../input/ieee-fraud-detection/'
CONN_STR = 'mysql+pymysql://root:mysqlpass@localhost:3306/fraud?charset=utf8'
# carefully handling mysql connection in multiprocess env.
# here, i create a connection for each process, store them in a dict.
PID_CONN = {}


def get_conn(conn_str):
    """get mysql connection for this process, if it dose not exists, create it."""
    pid = os.getpid()
    if pid not in PID_CONN.keys():
        PID_CONN[pid] = {conn_str: create_engine(conn_str)}
    else:
        if conn_str not in PID_CONN[pid].keys():
            PID_CONN[pid][conn_str] = create_engine(conn_str)

    return PID_CONN[pid][conn_str]


class Dump2MySQL():

    def __init__(self, conn_str, index=True, index_label: str = None, if_exists='fail'):
        self.conn_str = conn_str
        self.index = index
        self.index_label = index_label
        self.if_exists = if_exists
        # set in dump_using_mp()
        self.table = None

    def dump_using_mp(self, csv_name, chunksize=1000):
        print("Now dumping: " + csv_name)
        df_chunks = pd.read_csv(os.path.join(DS_DIR, csv_name + '.csv'), chunksize=chunksize)
        self.table = csv_name
        # dump some data to mysql in main process to create table,
        # otherwise, sub-processes will compete creating table, which may cause error.
        self._dump_df(df_chunks.get_chunk(size=chunksize))
        with Pool() as pool:
            _ = list(tqdm(pool.imap(self._dump_df, df_chunks, chunksize=1)))
        return

    def _dump_df(self, df):
        pd.io.sql.to_sql(df,
                         name=self.table,
                         con=get_conn(self.conn_str),
                         index=self.index,
                         index_label=self.index_label,
                         if_exists=self.if_exists)


dumper = Dump2MySQL(conn_str=CONN_STR, index=False, index_label='TransactionID', if_exists='append')

dumper.dump_using_mp('train_identity')
dumper.dump_using_mp('train_transaction')

dumper.dump_using_mp('test_identity')
dumper.dump_using_mp('test_transaction')
dumper.dump_using_mp('sample_submission')





