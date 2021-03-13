#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"

# _______________________________________________________________________________________
# imports:
import json
from datetime import timedelta
from functools import partial

from fraud_utils import *

# _______________________________________________________________________________________
# configs:
DS_DIR = eval('DS_DIR')
FILLNA_BY = -1

# seed_everything(SEED)
SEED = 42
LOCAL_TEST = False
TARGET = 'isFraud'

START_DATE = datetime.strptime('2017-11-30', '%Y-%m-%d')
UNIX_EPOCH = datetime.strptime('1970-01-01', '%Y-%m-%d')

# _______________________________________________________________________________________
# load dataset:
print('Loading Data...')
train_merge: pd.DataFrame = pd.read_pickle(os.path.join(DS_DIR, 'train_merge.pkl'))
test_merge: pd.DataFrame = pd.read_pickle(os.path.join(DS_DIR, 'test_merge.pkl'))
# align columns.
test_merge = test_merge[train_merge.columns]

# LABEL ENCODE AND MEMORY REDUCE
for i,f in enumerate(train_merge.columns):
    # FACTORIZE CATEGORICAL VARIABLES
    if (np.str(train_merge[f].dtype)=='category')|(train_merge[f].dtype=='object'):
        pass
        # df_comb = pd.concat([train_merge[f],test_merge[f]],axis=0)
        # df_comb,_ = df_comb.factorize(sort=True)
        # if df_comb.max()>32000: print(f,'needs int32')
        # train_merge[f] = df_comb[:len(train_merge)].astype('int16')
        # test_merge[f] = df_comb[len(train_merge):].astype('int16')
    # SHIFT ALL NUMERICS POSITIVE. SET NAN to -1
    elif f not in ['TransactionAmt','TransactionDT']:
        mn = np.min((train_merge[f].min(),test_merge[f].min()))
        train_merge[f] -= np.float32(mn)
        test_merge[f] -= np.float32(mn)
        train_merge[f].fillna(FILLNA_BY,inplace=True)
        test_merge[f].fillna(FILLNA_BY,inplace=True)


# _______________________________________________________________________________________
# FEOperator:
def fillna(*feat: pd.Series, fillna_by=FILLNA_BY):
    ret = tuple(map(lambda f: f.fillna(fillna_by), pack_singleton(feat)))
    return unpack_singleton(ret)


def self_freq_enc(feat: pd.Series, na_action=None) -> pd.Series:
    vc = feat.value_counts()
    return feat.map(arg=vc, na_action=na_action)


def self_label_enc(feat: pd.Series, na_action=None) -> pd.Series:
    _, uniques = feat.factorize()
    return feat.map(arg=pd.Series(range(len(uniques)), index=uniques), na_action=na_action)


def concat_feat(df1, df2, feat: str) -> pd.Series:
    return pd.concat([df1[feat], df2[feat]])


def interact(*feats: pd.Series, sep='_') -> pd.Series:
    from functools import reduce

    def join_ser(x: pd.Series, y: pd.Series):
        return x.astype(str) + sep + y.astype(str)

    return reduce(join_ser, feats)


def concat_interact(df1, df2, feat1: str, feat2: str) -> pd.Series:
    f1: pd.Series = concat_feat(df1, df2, feat1)
    f2: pd.Series = concat_feat(df1, df2, feat2)
    return interact(f1, f2)


def make_fe_desc_of_interact_label_enc(feat1: str, feat2: str, dst_as=None):
    return FEDesc((feat1, feat2),
                  op=[interact, TgtMeanEnc(tgt=train_merge[TARGET], by=train_merge[''])],
                  # op=InteractLabelEnc(fit_on=C_I_TR_TE(feat1, feat2)),
                  dst=feat1 + '__' + feat2,
                  astype=dst_as)


def delta_to_datetime(feat: pd.Series) -> pd.Series:
    return feat.map(lambda x: (START_DATE + timedelta(seconds=x)))


def get_DT_M(feat: pd.Series) -> pd.Series:
    tmp = delta_to_datetime(feat)
    return (tmp.dt.year - 2017) * 12 + tmp.dt.month


# V COLUMNS TO LOAD DECIDED BY CORRELATION EDA
# https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id
v =  [1, 3, 4, 6, 8, 11]
v += [13, 14, 17, 20, 23, 26, 27, 30]
v += [36, 37, 40, 41, 44, 47, 48]
v += [54, 56, 59, 62, 65, 67, 68, 70]
v += [76, 78, 80, 82, 86, 88, 89, 91]

#v += [96, 98, 99, 104] #relates to groups, no NAN
v += [107, 108, 111, 115, 117, 120, 121, 123] # maybe group, no NAN
v += [124, 127, 129, 130, 136] # relates to groups, no NAN

# LOTS OF NAN BELOW
v += [138, 139, 142, 147, 156, 162] #b1
v += [165, 160, 166] #b1
v += [178, 176, 173, 182] #b2
v += [187, 203, 205, 207, 215] #b2
v += [169, 171, 175, 180, 185, 188, 198, 210, 209] #b2
v += [218, 223, 224, 226, 228, 229, 235] #b3
v += [240, 258, 257, 253, 252, 260, 261] #b3
v += [264, 266, 267, 274, 277] #b3
v += [220, 221, 234, 238, 250, 271] #b3

v += [294, 284, 285, 286, 291, 297] # relates to grous, no NAN
v += [303, 305, 307, 309, 310, 320] # relates to groups, no NAN
v += [281, 283, 289, 296, 301, 314] # relates to groups, no NAN
#v += [332, 325, 335, 338] # b4 lots NAN

# #### feature summary:
# CREATE TABLE `train_transaction` (
#   `TransactionID`     -- PRIMARY KEY
#   `isFraud`           -- TARGET
#   `TransactionDT`
#   `TransactionAmt`
#   `ProductCD`         --                  category
#   `card1` - `card6`                       category
#   `addr1` - `addr2`   -- type number      category
#   `dist1` - `dist2`                       
#   `P_emaildomain`                         category
#   `R_emaildomain`     -- text,            category
#   `C1` - `C14`        -- counting         
#   `D1` - `D15`        -- time delta       
#   `M1` - `M9`         -- "T", "F"...      category
#   `V1` - `V339`       -- versa FEOperator
# ) ENGINE=InnoDB DEFAULT CHARSET=utf8;
#
# CREATE TABLE `train_identity` (
#   `TransactionID`                             -- PRIMARY KEY
#   `id_01` - `id_11`   double DEFAULT NULL,
#   `id_12` - `id_12`   text,
#   `id_13` - `id_14`   double DEFAULT NULL,    -- category
#   `id_15` - `id_16`   text,                   -- category
#   `id_17` - `id_22`   double DEFAULT NULL,    -- category
#   `id_23` - `id_23`   text,                   -- category
#   `id_24` - `id_26`   double DEFAULT NULL,    -- category
#   `id_27` - `id_31`   text,                   -- category
#   `id_32` - `id_32`   double DEFAULT NULL,    -- category
#   `id_33` - `id_38`   text,                   -- category
#   `DeviceType`        text,                   -- category
#   `DeviceInfo`        text,                   -- category
#   PRIMARY KEY (`TransactionID`)
# ) ENGINE=InnoDB DEFAULT CHARSET=utf8;


def desc(dst, op, src=None, astype=None, flag: FEFlag=FEFlag.T):
    src = dst if src is None else src
    return FEDesc(src, op, dst, astype, flag)


train_and_test = pd.concat([train_merge, test_merge])


def train_and_test_feat(feat: str):
    return train_and_test[feat]


# concat train and test (for same feature)
# C_TR_TE = partial(concat_feat, train_merge, test_merge)
C_TR_TE = train_and_test_feat
# concat then interact train and test (for feat1 and feat2)
C_I_TR_TE = partial(concat_interact, train_merge, test_merge)
# interact feat1, feat2 in train
I_TR = partial(concat_interact, train_merge, None)

FE_DESC_OF_INTERACT_LE = make_fe_desc_of_interact_label_enc

SELF_FREQ_ENC_NORMAL = partial(self_freq_enc, normalize=True)


# do_fe(train_and_test, inplace=True, fe_desc_table=[
#
#
#     # user-defined FE: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     # desc('cents', lambda f: (f - np.floor(f)), src='TransactionAmt', astype='float32'),
#
#     # desc('card1_addr1', [interact, self_label_enc], src=['card1', 'addr1']),
#     # desc('card1_addr1_P_emaildomain', [interact, self_label_enc], src=['card1', 'addr1', 'P_emaildomain']),
#
#     # desc('unix_ts', lambda x: (delta_to_datetime(x) - UNIX_EPOCH).map(lambda d: d.total_seconds()),
#     #      src='TransactionDT', astype='int32'),
#     # desc('day', lambda f: f // (24*60*60), src='TransactionDT'),
#     # desc('D1n', lambda day, D1: day - D1, src=['day', 'D1']),
#     # desc('uid', interact, src=['card1', 'addr1', 'D1n']),
#
#     # desc('DT_M', get_DT_M, src='TransactionDT'),
#
#     # original FE: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     # desc('P_emaildomain', self_label_enc, ),
#     # desc('R_emaildomain', self_label_enc, ),
#
# ])
# fe_concat = do_fe(pd.concat([train_merge, test_merge]), fe_desc_table=merge_fe_desc_table)
# train_merge_fe, test_merge_fe = fe_concat[: len(train_merge)], fe_concat[len(train_merge) :]


str_type = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain','M1', 'M2', 'M3', 'M4','M5',
            'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30',
            'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']

merge_fe_desc_table = [
    # -------------------------------- Transaction --------------------------------
    desc('isFraud', NOOP, ),
    desc('TransactionAmt', NOOP, ),
    # user-defined FE: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    desc('DT_M', get_DT_M, src='TransactionDT', flag=FEFlag.TB),
    desc('cents', lambda f: (f - np.floor(f)), src='TransactionAmt', astype='float32', flag=FEFlag.TB),
    desc('unix_ts', lambda x: (delta_to_datetime(x) - UNIX_EPOCH).map(lambda d: d.total_seconds()),
         src='TransactionDT', astype='int32', flag=FEFlag.B),
    desc('day', lambda f: f // (24 * 60 * 60), src='TransactionDT', flag=FEFlag.B),
    desc('D1n', lambda day, D1: day - D1, src=['day', 'D1'], flag=FEFlag.B),
    desc('uid', interact, src=['card1', 'addr1', 'D1n'], flag=FEFlag.B),

    desc('ProductCD', self_label_enc, astype='category'),  # dst_as='category'),

    desc('card1', NOOP, src='card1'),
    desc('card2', NOOP, src='card2'),
    desc('card3', NOOP, src='card3'),
    # desc('card4', FreqEnc(count_on=C_TR_TE('card4')), astype=None),
    desc('card5', NOOP, ),
    desc('card6', self_label_enc, astype='category'),

    desc('addr1', NOOP, ),
    desc('addr2', NOOP, ),

    desc('dist1', NOOP, ),
    desc('dist2', NOOP, ),

    desc('P_emaildomain', self_label_enc, astype='category', flag=FEFlag.TB),
    desc('R_emaildomain', self_label_enc, astype='category', flag=FEFlag.TB),

    # user-defined FE: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    desc('card1_FE', self_freq_enc, src='card1'),
    desc('card2_FE', self_freq_enc, src='card2'),
    desc('card3_FE', self_freq_enc, src='card3'),
    desc('addr1_FE', self_freq_enc, src='addr1'),
    desc('P_emaildomain_FE', self_freq_enc, src='P_emaildomain'),
    desc('card1_addr1', [interact, self_label_enc], src=['card1', 'addr1'], flag=FEFlag.TB),
    desc('card1_addr1_P_emaildomain', [interact, self_label_enc], src=['card1', 'addr1', 'P_emaildomain'], flag=FEFlag.TB),
    desc('card1_addr1_FE', self_freq_enc, src='card1_addr1'),
    desc('card1_addr1_P_emaildomain_FE', self_freq_enc, src='card1_addr1_P_emaildomain'),

    *[desc(f'C{i}', NOOP, ) for i in range(1, 14 + 1) if f'C{i}' not in ['C3', ]],
    *[desc(f'D{i}', NOOP, ) for i in range(1, 15 + 1) if f'D{i}' not in ['D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14']],

    desc('M1', self_label_enc, flag=FEFlag.TB),
    desc('M2', self_label_enc, flag=FEFlag.TB),
    desc('M3', self_label_enc, flag=FEFlag.TB),
    desc('M4', self_label_enc, flag=FEFlag.TB),
    desc('M5', self_label_enc, flag=FEFlag.B),
    desc('M6', self_label_enc, flag=FEFlag.TB),
    desc('M7', self_label_enc, flag=FEFlag.TB),
    desc('M8', self_label_enc, flag=FEFlag.TB),
    desc('M9', self_label_enc, flag=FEFlag.TB),

    *[desc(f'V{i}', NOOP, ) for i in range(1, 339 + 1) if i in v],

    # -------------------------------- Identity --------------------------------
    # ['card4','id_07','id_14','id_21','id_30','id_32','id_34']
    *[desc(f'id_{i:02d}', NOOP, ) for i in range(1, 11 + 1) if f'id_{i:02d}' not in ['id_07', 'id_08']],

    desc('id_12', self_label_enc, astype='category'),
    desc('id_13', NOOP, ),  # astype='category'),
    # desc('id_14', NOOP, ),  # dst_as='category'),
    desc('id_15', self_label_enc, astype='category'),
    desc('id_16', self_label_enc, astype='category'),
    desc('id_17', NOOP, ),  # astype='category'),
    desc('id_18', NOOP, ),  # astype='category'),
    desc('id_19', NOOP, ),  # astype='category'),
    desc('id_20', NOOP, ),  # astype='category'),
    # desc('id_21', NOOP, astype='category'),
    # desc('id_22', NOOP, astype='category'),
    # desc('id_23', MapEnc({'TRANSPARENT': 4, 'IP_PROXY': 3,
    #                      'IP_PROXY:ANONYMOUS': 2, 'IP_PROXY:HIDDEN': 1}), astype='category'),
    # desc('id_24', NOOP, astype='category'),
    # desc('id_25', NOOP, astype='category'),
    # desc('id_26', NOOP, astype='category'),
    # desc('id_27', MapEnc({'Found': 1, 'NotFound': 0}), astype='category'),
    desc('id_28', self_label_enc, astype='category'),
    desc('id_29', self_label_enc, astype='category'),
    # desc('id_30', FreqEnc(count_on=C_TR_TE('id_30')), ),  # dst_as='category'),
    desc('id_31', self_label_enc, astype='category'),
    # desc('id_32', NOOP, astype='category'),
    # desc('id_33', FreqEnc(count_on=C_TR_TE('id_33')), astype='category'),
    # desc('id_34', RegexEnc(r'.*:([-+]?\d+)', do_eval=True), ),  # dst_as='category'),
    desc('id_35', self_label_enc, astype='category'),
    desc('id_36', self_label_enc, astype='category'),
    desc('id_37', self_label_enc, astype='category'),
    desc('id_38', self_label_enc, astype='category'),

    desc('DeviceType', self_label_enc, astype='category'),
    desc('DeviceInfo', self_label_enc, astype='category'),

    # # user-defined FE: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # encode_AG(['TransactionAmt','D9','D11'],['card1','card1_addr1','card1_addr1_P_emaildomain'],['mean','std'],usena=True)
    *[
        desc(f'{tgt}_{by}_{agg}', [fillna, GroupFE(agg=agg)], src=[tgt, by])
        for agg in ['mean', 'std']
        for by in ['card1','card1_addr1','card1_addr1_P_emaildomain']
        for tgt in ['TransactionAmt', 'D9', 'D11']
    ],

    desc('uid_FE', self_freq_enc, src='uid'),

    *[
        desc(f'{tgt}_uid_{agg}', [fillna, GroupFE(agg=agg), fillna], src=[tgt, 'uid'])
        for agg in ['mean', 'std']
        for tgt in ['TransactionAmt', 'D4', 'D9', 'D10', 'D15']
    ],

    *[desc(f'{tgt}_uid_mean', [fillna, GroupFE(agg='mean'), fillna], src=[tgt, 'uid'])
      for tgt in ['C' + str(x) for x in range(1, 15) if x != 3]],

    *[desc(f'{tgt}_uid_mean', [fillna, GroupFE(agg='mean'), fillna], src=[tgt, 'uid'])
      for tgt in ['M' + str(x) for x in range(1, 10)]],

    *[desc(f'uid_{tgt}_ct', GroupFE(agg='nunique'), src=[tgt, 'uid'])
      for tgt in ['P_emaildomain', 'dist1', 'DT_M', 'id_02', 'cents']],

    *[desc(f'{tgt}_uid_std', [fillna, GroupFE(agg='std'), fillna], src=[tgt, 'uid'])
      for tgt in ['C14', ]],

    *[desc(f'uid_{tgt}_ct', GroupFE(agg='nunique'), src=[tgt, 'uid'])
      for tgt in ['C13', 'V314', 'V127','V136','V309','V307','V320']],

    desc('outsider15', lambda D1, D15: np.abs(D1 - D15) > 3, src=['D1', 'D15'], astype='int8'),
]
fe_concat = do_fe(train_and_test, fe_desc_table=merge_fe_desc_table)
train_merge_fe, test_merge_fe = fe_concat[: len(train_merge)], fe_concat[len(train_merge) :]

feat_columns = [x for x in train_merge_fe.columns if x not in ['TransactionID', 'TransactionDT', 'DT_M', TARGET]]
merge_cate_feat = [desc.dst for desc in merge_fe_desc_table if desc.astype == 'category']

# _______________________________________________________________________________________
# dump outputs:
train_merge_fe.to_pickle('train_merge.pkl')
test_merge_fe.to_pickle('test_merge.pkl')

config = {
    'feat_columns': feat_columns,
    'merge_cate_feat': merge_cate_feat,
    'fillna_by': FILLNA_BY,
}

with open('config.json', 'w') as f:
    json.dump(config, f)

