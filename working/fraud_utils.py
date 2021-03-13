#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"


import abc
import os
import random
import re
from contextlib import contextmanager
from datetime import datetime
from enum import IntFlag
from typing import *

import numpy as np
import pandas as pd


# _______________________________________________________________________________________
# util functions:
def pack_singleton(x):
    return x if isinstance(x, (List, Tuple)) else tuple([x, ])


def unpack_singleton(x):
    return x[0] if len(x) == 1 else tuple(x)


def self_freq_enc(ser: pd.Series, normalize=False):
    vc = ser.value_counts()
    map_ser: pd.Series = vc / len(ser) if normalize else vc
    return ser.map(map_ser)


# _______________________________________________________________________________________
# classes:
class FEFlag(IntFlag):
    """T: WriteThrough; B: WriteBack; TB: WriteThrough and WriteBack"""
    T = 1  # WriteThrough
    B = 2  # WriteBack
    TB = T | B  # WriteThrough and WriteBack


class FEOperator:
    """Feature Engineering Base class."""
    @abc.abstractmethod
    def __call__(self, *feat: pd.Series) -> Union[pd.Series, Tuple[pd.Series]]: ...


class FEDesc:
    def __init__(self, src, op, dst=None, astype=None, flag: FEFlag = FEFlag.T):
        """
        describe FE process: source feature -> feature operation -> destination feature.
        and the destination feature's properties.

        :param src: string or list/tuple of strings(means multi features as input of operation);
        :param op: operation or list of operations(means pipeline of those operations)
                   operation could be FEOperator or Callable[[*pd.Series], *pd.Series] ;
        :param dst: string or list/tuple of strings(means multi features as output of operation);
        :param astype: string: specify dst's dtype.
        """
        self.src = src
        self.op = op
        self.dst = dst if dst is not None else src
        self.astype = astype
        self.flag = flag


class _NOOP(FEOperator):
    """Bypass features."""
    def __call__(self, *feat: pd.Series) -> Union[pd.Series, Tuple[pd.Series]]:
        return feat


NOOP = _NOOP()


class GroupFE(FEOperator):
    """GroupBy 'by' feature, and do 'agg' functor within group"""
    def __init__(self, agg):
        self.agg = agg

    def __call__(self, *feat: pd.Series) -> Union[pd.Series, Tuple[pd.Series]]:
        assert len(feat) == 2, 'GroupFE expect 2 input by now.'
        tgt, by = feat
        return tgt.groupby(by).transform(self.agg)


# _______________________________________________________________________________________
# #### Encoding classes:
class EncodeFeat(FEOperator):
    """Feature encoder."""
    def __call__(self, *feat: pd.Series) -> Union[pd.Series, Tuple[pd.Series]]:
        return self.encode_feat(*feat)

    @abc.abstractmethod
    def encode_feat(self, *feat: pd.Series) -> Union[pd.Series, Tuple[pd.Series]]: ...


class MapEnc(EncodeFeat):
    """Map encoding."""
    def __init__(self, map_arg, na_action=None):
        self.map_arg = map_arg
        self.na_action = na_action

    def encode_feat(self, *feat: pd.Series) -> pd.Series:
        assert len(feat) == 1, 'MapEnc handle feature one by one, but got multi features!'
        return feat[0].map(arg=self.map_arg, na_action=self.na_action)

    def __str__(self):
        return f'map encoding'


class FreqEnc(MapEnc):
    """Frequency encoding."""
    def __init__(self, count_on: pd.Series, normalize=False):
        vc = count_on.value_counts()
        map_ser: pd.Series = vc / len(count_on) if normalize else vc
        super().__init__(map_arg=map_ser)

    def __str__(self):
        return 'frequency encoding'


class RegexEnc(MapEnc):
    """Regular expression encoding."""
    def __init__(self, regex, do_eval=True):
        self.regex = regex
        if do_eval:
            arg = lambda s: eval(re.match(regex, s).group(1))
        else:
            arg = lambda s: re.match(regex, s).group(1)

        super().__init__(map_arg=arg, na_action='ignore')

    def __str__(self):
        return 'regular expression encoding'


class TgtMeanEnc(MapEnc):
    """Target mean encoding."""
    def __init__(self, tgt: pd.Series, by: pd.Series):
        self.group_mean: pd.Series = tgt.groupby(by).mean()
        super().__init__(map_arg=self.group_mean, na_action='ignore')

    def __str__(self):
        return 'target mean encoding'


class LabelEnc(MapEnc):
    """Label encoding."""
    def __init__(self, fit_on: pd.Series, na_action=None):
        _, uniques = fit_on.factorize()
        super().__init__(map_arg=pd.Series(range(len(uniques)), index=uniques), na_action=na_action)

    def __str__(self):
        return 'label encoding'


# _______________________________________________________________________________________
# functions.
# _______________________________________________________________________________________
# #### start feature engineering:
def do_fe(*df_on: pd.DataFrame, fe_desc_table: List[FEDesc], inplace=False):
    """ do feature operations registered in feat_op_table on df. """
    ret = []
    for df_on_elem in df_on:
        # #### handle one df.
        ret_elem = pd.DataFrame()
        for desc in fe_desc_table:
            # #### handle one fe-descriptor, e.g. one pipeline of operations.
            mediate_res = tuple([df_on_elem[feat].copy() for feat in pack_singleton(desc.src)])
            for op in pack_singleton(desc.op):
                # #### handle one operator in pipeline.
                if callable(op):
                    if not isinstance(op, _NOOP):
                        print(f'Handling feature {desc.src} -> {desc.dst} using {op} ...')
                    mediate_res = pack_singleton(op(*mediate_res))
                else:
                    print('Illegal feature operation! continue ...')

            if FEFlag.T in desc.flag:
                for dst, col in zip(pack_singleton(desc.dst), mediate_res):
                    ret_elem[dst] = col if desc.astype is None else col.astype(desc.astype)
            if FEFlag.B in desc.flag:
                for dst, col in zip(pack_singleton(desc.dst), mediate_res):
                    df_on_elem[dst] = col if desc.astype is None else col.astype(desc.astype)

        ret.append(ret_elem)

    return unpack_singleton(ret)


# _______________________________________________________________________________________
# #### helpers:
@contextmanager
def timer(name):
    """
    Time Each Process
    """
    tic = datetime.now()
    yield
    toc = datetime.now()
    print(">"*30 + name + " spend: %dm %.3fs" % divmod((toc - tic).total_seconds(), 60))


def df_stats(df: pd.DataFrame, top_n=5):
    total = len(df)
    stats = pd.DataFrame(index=df.columns, columns=['dtype', 'distinct-cnt', 'non-null-cnt'])

    for col_name, series in df.iteritems():
        series_dtype = series.dtype
        series_value_counts = series.value_counts()
        dist_cnt = len(series_value_counts)
        non_null_cnt = series.count()

        stats.loc[col_name] = [series_dtype, dist_cnt, non_null_cnt]

        name_str = f"{col_name}({series_dtype})".rjust(25, '-')
        dist_cnt_str = str(dist_cnt).rjust(6, ' ')
        non_null_cnt_str = str(non_null_cnt).rjust(6, ' ')

        print(f"{name_str} : count distinct - {dist_cnt_str} : "
              f"non-null/total - {non_null_cnt_str}/{total} = {non_null_cnt/total:.3f} ")
        if top_n > 0:
            col_value_count_list = [
                "'" + str(c) + "'" + ":" + str(n) for c, n in sorted(
                    series_value_counts.items(),
                    key=lambda kv: kv[1],
                    reverse=True
                )
            ]
            print(", ".join(col_value_count_list[:min(len(col_value_count_list), top_n)]))

    stats['null-cnt'] = total - stats['non-null-cnt']
    stats['non-null-ratio'] = stats['non-null-cnt'] / total
    stats['total'] = total
    return stats


# _______________________________________________________________________________________
# #### seed functions.
def seed_everything(seed=0):
    """seed to make all processes deterministic"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# _______________________________________________________________________________________
# #### reduce memory usage by downcast int* and float*. this is inplace operation.
def reduce_mem_usage(df, deep=True, verbose=True, categories=False) -> None:

    def memory_usage_mb(df, *args, **kwargs):
        """Dataframe memory usage in MB. """
        return df.memory_usage(*args, **kwargs).sum() / 1024 ** 2

    # All types that we want to change for "lighter" ones.
    # int8 and float16 are not include because we cannot reduce
    # those data types.
    # float32 is not include because float16 has too low precision.
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = memory_usage_mb(df, deep=deep)

    for col, col_type in df.dtypes.iteritems():
        best_type = None
        if categories and col_type == "object":
            df[col] = df[col].astype("category")
            best_type = "category"
        elif col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
            best_type = df[col].dtype.name
        # Log the conversion performed.
        if verbose and best_type is not None and best_type != str(col_type):
            print(f"Column '{col}' converted from {col_type} to {best_type}")

    if verbose:
        end_mem = memory_usage_mb(df, deep=deep)
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(f"Memory usage decreased from"
              f" {start_mem:.2f}MB to {end_mem:.2f}MB"
              f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")



