#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"

# _______________________________________________________________________________________
# imports:
import json
from functools import partial

from fraud_utils import *

# _______________________________________________________________________________________
# configs:
DS_DIR = eval('DS_DIR')

SEED = 42
# seed_everything(SEED)
LOCAL_TEST = False
TARGET = 'isFraud'

# _______________________________________________________________________________________
# load dataset:
print('Loading Data...')
train_merge: pd.DataFrame = pd.read_pickle(os.path.join(DS_DIR, 'train_merge.pkl'))
test_merge: pd.DataFrame = pd.read_pickle(os.path.join(DS_DIR, 'test_merge.pkl'))
# align columns.
test_merge = test_merge[train_merge.columns]


# _______________________________________________________________________________________
# FEOperator:
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


def desc(dst, op, src=None, astype=None):
    src = dst if src is None else src
    return FEDesc(src, op, dst, astype)


# concat train and test (for same feature)
C_TR_TE = partial(concat_feat, train_merge, test_merge)
# concat then interact train and test (for feat1 and feat2)
C_I_TR_TE = partial(concat_interact, train_merge, test_merge)
# interact feat1, feat2 in train
I_TR = partial(concat_interact, train_merge, None)

FE_DESC_OF_INTERACT_LE = make_fe_desc_of_interact_label_enc

SELF_FREQ_ENC_NORMAL = partial(self_freq_enc, normalize=True)

do_fe(train_merge, test_merge, inplace=True, fe_desc_table=[
    # user-defined FE: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    desc('card1_addr1', interact, src=['card1', 'addr1']),
    desc('card1_addr1_P_emaildomain', interact, src=['card1', 'addr1', 'P_emaildomain']),
])

# str_type = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain','M1', 'M2', 'M3', 'M4','M5',
#             'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30',
#             'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']
merge_fe_desc_table = [
    # -------------------------------- Transaction --------------------------------
    desc('isFraud', NOOP, ),
    # user-defined FE: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    desc('TransactionAmt', NOOP, ),
    desc('cents', lambda f: (f - np.floor(f)), src='TransactionAmt', astype='float32'),

    desc('ProductCD', LabelEnc(fit_on=C_TR_TE('ProductCD')), astype='category'),  # dst_as='category'),

    desc('card1', FreqEnc(count_on=C_TR_TE('card1')), ),
    desc('card2', FreqEnc(count_on=C_TR_TE('card2')), ),
    desc('card3', FreqEnc(count_on=C_TR_TE('card3')), ),
    # desc('card4', FreqEnc(count_on=C_TR_TE('card4')), astype=None),
    desc('card5', NOOP, ),
    desc('card6', MapEnc({'credit': 1, 'debit': 0}), astype='category'),
    # user-defined FE: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    desc('addr1', FreqEnc(count_on=C_TR_TE('addr1')), ),
    # desc('addr2', NOOP, ),

    desc('dist1', NOOP, ),
    desc('dist2', NOOP, ),

    desc('P_emaildomain', LabelEnc(fit_on=C_TR_TE('P_emaildomain'), ), astype='category'),
    # desc('R_emaildomain', LabelEnc(fit_on=C_TR_TE('R_emaildomain'), na_action='ignore'), astype='category'),

    *[desc(f'C{i}', NOOP, ) for i in range(1, 14 + 1) if i not in ['C3']],
    *[desc(f'D{i}', NOOP, ) for i in range(1, 15 + 1) if i not in ['D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14']],

    desc('M1', MapEnc({'T': 1, 'F': 0}), astype='category'),
    desc('M2', MapEnc({'T': 1, 'F': 0}), astype='category'),
    desc('M3', MapEnc({'T': 1, 'F': 0}), astype='category'),
    desc('M4', RegexEnc(r'M(\d*)', do_eval=True), astype='category'),
    # desc('M5', MapEnc({'T': 1, 'F': 0}), astype='category'),
    desc('M6', MapEnc({'T': 1, 'F': 0}), astype='category'),
    desc('M7', MapEnc({'T': 1, 'F': 0}), astype='category'),
    desc('M8', MapEnc({'T': 1, 'F': 0}), astype='category'),
    desc('M9', MapEnc({'T': 1, 'F': 0}), astype='category'),

    *[desc(f'V{i}', NOOP, ) for i in range(1, 339 + 1) if i in v],

    # -------------------------------- Identity --------------------------------
    # ['card4','id_07','id_14','id_21','id_30','id_32','id_34']
    *[desc(f'id_{i:02d}', NOOP, ) for i in range(1, 11 + 1) if i not in ['id_07', 'id_08']],

    desc('id_12', MapEnc({'Found': 1, 'NotFound': 0}), astype='category'),
    desc('id_13', NOOP, astype='category'),
    # desc('id_14', NOOP, ),  # dst_as='category'),
    desc('id_15', MapEnc({'New': 2, 'Found': 1, 'Unknown': 0}), astype='category'),
    desc('id_16', MapEnc({'Found': 1, 'NotFound': 0}), astype='category'),
    desc('id_17', NOOP, astype='category'),
    desc('id_18', NOOP, astype='category'),
    desc('id_19', NOOP, astype='category'),
    desc('id_20', NOOP, astype='category'),
    # desc('id_21', NOOP, astype='category'),
    # desc('id_22', NOOP, astype='category'),
    # desc('id_23', MapEnc({'TRANSPARENT': 4, 'IP_PROXY': 3,
    #                      'IP_PROXY:ANONYMOUS': 2, 'IP_PROXY:HIDDEN': 1}), astype='category'),
    # desc('id_24', NOOP, astype='category'),
    # desc('id_25', NOOP, astype='category'),
    # desc('id_26', NOOP, astype='category'),
    # desc('id_27', MapEnc({'Found': 1, 'NotFound': 0}), astype='category'),
    desc('id_28', MapEnc({'New': 2, 'Found': 1}), astype='category'),
    desc('id_29', MapEnc({'Found': 1, 'NotFound': 0}), astype='category'),
    # desc('id_30', FreqEnc(count_on=C_TR_TE('id_30')), ),  # dst_as='category'),
    desc('id_31', FreqEnc(count_on=C_TR_TE('id_31')), ),  # dst_as='category'),
    # desc('id_32', NOOP, astype='category'),
    # desc('id_33', FreqEnc(count_on=C_TR_TE('id_33')), astype='category'),
    # desc('id_34', RegexEnc(r'.*:([-+]?\d+)', do_eval=True), ),  # dst_as='category'),
    desc('id_35', MapEnc({'T': 1, 'F': 0}), astype='category'),
    desc('id_36', MapEnc({'T': 1, 'F': 0}), astype='category'),
    desc('id_37', MapEnc({'T': 1, 'F': 0}), astype='category'),
    desc('id_38', MapEnc({'T': 1, 'F': 0}), astype='category'),

    desc('DeviceType', MapEnc({'desktop': 1, 'mobile': 0}), astype='category'),
    desc('DeviceInfo', FreqEnc(count_on=C_TR_TE('DeviceInfo'), ), astype='category'),

    # # user-defined FE: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    desc('card1_addr1_cnt', FreqEnc(count_on=C_TR_TE('card1_addr1')), src='card1_addr1'),
    desc('card1_addr1_P_emaildomain_cnt', FreqEnc(count_on=C_TR_TE('card1_addr1_P_emaildomain')), src='card1_addr1_P_emaildomain'),

    # encode_AG(['TransactionAmt','D9','D11'],['card1','card1_addr1','card1_addr1_P_emaildomain'],['mean','std'],usena=True)
    *np.array([
        [
            desc('_'.join([tgt, by, 'mean']), GroupFE(by=C_TR_TE(by), agg='mean'), src=tgt)
            for tgt in ['TransactionAmt','D9','D11']
        ]
        for by in ['card1','card1_addr1','card1_addr1_P_emaildomain']
    ]).flatten().tolist(),
    *np.array([
        [
            desc('_'.join([tgt, by, 'std']), [GroupFE(by=C_TR_TE(by), agg='std'), lambda f: f.fillna(0)], src=tgt)
            for tgt in ['TransactionAmt', 'D9', 'D11']
        ]
        for by in ['card1', 'card1_addr1', 'card1_addr1_P_emaildomain']
    ]).flatten().tolist(),

]

fe_concat = do_fe(pd.concat([train_merge, test_merge]), fe_desc_table=merge_fe_desc_table)
train_merge_fe, test_merge_fe = fe_concat[: len(train_merge)], fe_concat[len(train_merge) :]

# _______________________________________________________________________________________
# Drop some features:
# useful_feat_pre_fe = 'TransactionAmt\nProductCD\ncard1\ncard2\ncard3\ncard4\ncard5\ncard6\naddr1\ndist1' \
#                      '\nP_emaildomain\nR_emaildomain\nC1\nC2\nC4\nC5\nC6\nC7\nC8\nC9\nC10\nC11\nC12\nC13\nC14\nD1\nD2' \
#                      '\nD3\nD4\nD5\nD6\nD8\nD9\nD10\nD11\nD12\nD13\nD14\nD15\nM2\nM3\nM4\nM5\nM6\nM8\nM9\nV4\nV5\nV12' \
#                      '\nV13\nV19\nV20\nV30\nV34\nV35\nV36\nV37\nV38\nV44\nV45\nV47\nV53\nV54\nV56\nV57\nV58\nV61\nV62' \
#                      '\nV70\nV74\nV75\nV76\nV78\nV82\nV83\nV87\nV91\nV94\nV96\nV97\nV99\nV126\nV127\nV128\nV130\nV131' \
#                      '\nV139\nV143\nV149\nV152\nV160\nV165\nV170\nV187\nV189\nV201\nV203\nV204\nV207\nV208\nV209' \
#                      '\nV210\nV212\nV217\nV221\nV222\nV234\nV257\nV258\nV261\nV264\nV265\nV266\nV267\nV268\nV271' \
#                      '\nV274\nV275\nV277\nV278\nV279\nV280\nV282\nV283\nV285\nV287\nV289\nV291\nV292\nV294\nV306' \
#                      '\nV307\nV308\nV310\nV312\nV313\nV314\nV315\nV317\nV323\nV324\nV332\nV333\nid_01\nid_02\nid_05' \
#                      '\nid_06\nid_09\nid_13\nid_14\nid_17\nid_19\nid_20\nid_30\nid_31\nid_33\nid_38\nDeviceType' \
#                      '\nDeviceInfo'
#
# useful_feat_pre_fe = [x for x in useful_feat_pre_fe.split('\n')]
# feat_to_drop = [x for x in train_merge.columns if x not in [*useful_feat_pre_fe, TARGET]]
#
# train_merge_fe.drop(columns=feat_to_drop, inplace=True, errors='ignore')
# test_merge_fe.drop(columns=feat_to_drop, inplace=True, errors='ignore')

merge_cate_feat = [desc.dst for desc in merge_fe_desc_table
                   if desc.astype == 'category']  # and desc.dst not in feat_to_drop]

# _______________________________________________________________________________________
# dump outputs:
train_merge_fe.to_pickle('train_merge.pkl')
test_merge_fe.to_pickle('test_merge.pkl')

with open('merge_cate_feat.json', 'w') as f:
    json.dump(merge_cate_feat, f)

