#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"

# _______________________________________________________________________________________
# imports:
import gc
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


def interact(feat1: pd.Series, feat2: pd.Series, sep='_') -> pd.Series:
    # noinspection PyUnresolvedReferences
    return feat1.astype(str) + sep + feat2.astype(str)


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


def standardize(ser: pd.Series):
    if abs(ser.std()) < 1e-3:
        return 0
    else:
        return (ser - ser.mean()) / ser.std()


def rank_with_equal_number(ser: pd.Series, rank=8):
    sorted = ser.sort_values()
    step = len(ser) / rank
    split_indexes = [int(x*step) for x in range(1, rank)]
    thresholds = np.array(sorted.iloc[split_indexes])
    relation_table = np.array(ser)[..., np.newaxis] > thresholds
    return pd.Series(relation_table.sum(axis=-1), index=ser.index)


def id_split(df):
    tmp = df['DeviceInfo'].str.replace(':', '/').str.split('/', expand=True)
    df['device_name'] = tmp[0]
    df['device_version'] = tmp[1]

    df['OS_id_30'] = df['id_30'].str.split(' ', expand=True)[0]
    df['version_id_30'] = df['id_30'].str.split(' ', expand=True)[1]

    tmp = df['id_31'].str.split(r' for ', expand=True)
    browser_type = tmp[0].str.extract(r'(.+?) [\d\.]+.*|(.*)')
    df['browser_type'] = np.where(browser_type[0].isna(), browser_type[1], browser_type[0])
    df['browser_ver'] = tmp[0].str.extract(r'.+? ([\d\.]+).*')
    df['browser_for'] = tmp[1]

    df['screen_width'] = df['id_33'].str.split('x', expand=True)[0].astype('float16')
    df['screen_height'] = df['id_33'].str.split('x', expand=True)[1].astype('float16')

    df.loc[df['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    df.loc[df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    df.loc[df['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    df.loc[df['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    df.loc[df['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    df.loc[df['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    df.loc[df.device_name.isin(df.device_name.value_counts()[
                                   df.device_name.value_counts() < 200].index), 'device_name'] = "Others"
    df['had_id'] = 1
    gc.collect()

    return df


train_merge = id_split(train_merge)
test_merge = id_split(test_merge)

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
    FEDesc('TransactionAmt', rank_with_equal_number,
           dst='TransactionAmt_rank'),
    # user-defined FE: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # FEDesc('DeviceInfo', lambda s: s.str.replace(':', '/').str.split('/', expand=True)[0],
    #        dst='dev_name'),
    # FEDesc('DeviceInfo', lambda s: s.str.replace(':', '/').str.split('/', expand=True)[1],
    #        dst='dev_ver'),
    # FEDesc('id_33', RegexEnc(r'(\d*)x*', do_eval=True), dst='id_33_W'),
    # FEDesc('id_33', RegexEnc(r'.*x(\d*)', do_eval=True), dst='id_33_H'),
    #
    # FEDesc('id_31', RegexEnc(r'(.+?) [\d\.]+.*|(.*)', ), dst='browser_type'),
    # FEDesc('id_31', RegexEnc(r'.+? ([\d\.]+).*', ), dst='browser_version'),
    # FEDesc('id_31', RegexEnc(r'.* for (.*)', ), dst='browser_for'),
    # user-defined FE: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    FEDesc(('id_02' 	   , 'id_20'         ,), interact, dst='id_02__id_20'),
    FEDesc(('id_02' 	   , 'D8'            ,), interact, dst='id_02__D8'),
    FEDesc(('card5' 	   , 'P_emaildomain' ,), interact, dst='card5__P_emaildomain'),
    FEDesc(('card2' 	   , 'id_20'         ,), interact, dst='card2__id_20'),
    FEDesc(('card2' 	   , 'dist1'         ,), interact, dst='card2__dist1'),
    FEDesc(('card1' 	   , 'card5'         ,), interact, dst='card1__card5'),
    FEDesc(('addr1' 	   , 'card1'         ,), interact, dst='addr1__card1'),
    FEDesc(('P_emaildomain', 'C2'            ,), interact, dst='P_emaildomain__C2'),
    FEDesc(('DeviceInfo'   , 'P_emaildomain' ,), interact, dst='DeviceInfo__P_emaildomain'),
    FEDesc(('D11' 	       , 'DeviceInfo'    ,), interact, dst='D11__DeviceInfo'),

    FEDesc('D9', lambda x: ~x.isna(), ),
])

merge_fe_desc_table = [
    # -------------------------------- Transaction --------------------------------
    # FEDesc('TransactionID', NOOP, ),
    FEDesc('isFraud', NOOP, ),
    # FEDesc('TransactionAmt', NOOP, ),
    # FEDesc('TransactionDT',  NOOP, ),
    # user-defined FE: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    FEDesc('TransactionAmt', lambda x: np.log(x),
           dst='TransactionAmt_log'),
    FEDesc('TransactionAmt_rank', NOOP, ),
    FEDesc('TransactionAmt', lambda x: x.astype(str).str.split('.', expand=True)[1].map(lambda s: len(s)),
           dst='TransactionAmt_precision'),
    FEDesc('TransactionDT', lambda x: np.floor(x / 3600).astype(int) % 24,
           dst='TransactionDT_hour', astype='category'),
    FEDesc('TransactionDT', lambda x: np.floor((x / (3600 * 24) - 1) % 7).astype(int),
           dst='TransactionDT_day_week', astype='category'),

    FEDesc('ProductCD', MapEnc({'W': 3, 'H': 2, 'C': 1, 'R': 0}), astype='category'),

    FEDesc('card1', NOOP, astype='category'),
    FEDesc('card2', NOOP, astype='category'),
    FEDesc('card3', NOOP, astype='category'),
    FEDesc('card4', MapEnc({'visa': 3, 'mastercard': 2, 'american express': 1, 'discover': 0}), astype='category'),
    FEDesc('card5', NOOP, astype='category'),
    FEDesc('card6', MapEnc({'credit': 1, 'debit': 0}), astype='category'),
    # user-defined FE: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # FEDesc('ProductCD', GroupFE(by=C_TR_TE('card1'), agg=SELF_FREQ_ENC_NORMAL),
    #        dst='ProductCD_freq_by_card1'),
    # FEDesc('TransactionAmt_rank', GroupFE(by=C_TR_TE('card1'), agg=SELF_FREQ_ENC_NORMAL),
    #        dst='TransactionAmt_rank_freq_by_card1'),
    # FEDesc('TransactionAmt', GroupFE(by=C_TR_TE('card1'), agg=standardize),
    #        dst='TransactionAmt_standardize_by_card1'),

    FEDesc('addr1', NOOP, astype='category'),
    FEDesc('addr2', NOOP, astype='category'),

    FEDesc('dist1', NOOP, ),
    FEDesc('dist2', NOOP, ),

    FEDesc('P_emaildomain', LabelEnc(fit_on=C_TR_TE('P_emaildomain'), na_action='ignore'), astype='category'),
    FEDesc('R_emaildomain', LabelEnc(fit_on=C_TR_TE('R_emaildomain'), na_action='ignore'), astype='category'),

    *[FEDesc(f'C{i}', NOOP, ) for i in range(1, 14 + 1)],
    *[FEDesc(f'D{i}', NOOP, ) for i in range(1, 15 + 1)],

    FEDesc('M1', MapEnc({'T': 1, 'F': 0}), astype='category'),
    FEDesc('M2', MapEnc({'T': 1, 'F': 0}), astype='category'),
    FEDesc('M3', MapEnc({'T': 1, 'F': 0}), astype='category'),
    FEDesc('M4', RegexEnc(r'M(\d*)', do_eval=True), astype='category'),
    FEDesc('M5', MapEnc({'T': 1, 'F': 0}), astype='category'),
    FEDesc('M6', MapEnc({'T': 1, 'F': 0}), astype='category'),
    FEDesc('M7', MapEnc({'T': 1, 'F': 0}), astype='category'),
    FEDesc('M8', MapEnc({'T': 1, 'F': 0}), astype='category'),
    FEDesc('M9', MapEnc({'T': 1, 'F': 0}), astype='category'),

    *[FEDesc(f'V{i}', NOOP, ) for i in range(1, 339 + 1)],

    # -------------------------------- Identity --------------------------------
    *[FEDesc(f'id_{i:02d}', NOOP, ) for i in range(1, 11 + 1)],

    FEDesc('id_12', MapEnc({'Found': 1, 'NotFound': 0}), astype='category'),
    FEDesc('id_13', NOOP, astype='category'),
    FEDesc('id_14', NOOP, ),  # dst_as='category'),
    FEDesc('id_15', MapEnc({'New': 2, 'Found': 1, 'Unknown': 0}), astype='category'),
    FEDesc('id_16', MapEnc({'Found': 1, 'NotFound': 0}), astype='category'),
    FEDesc('id_17', NOOP, astype='category'),
    FEDesc('id_18', NOOP, astype='category'),
    FEDesc('id_19', NOOP, astype='category'),
    FEDesc('id_20', NOOP, astype='category'),
    FEDesc('id_21', NOOP, astype='category'),
    FEDesc('id_22', NOOP, astype='category'),
    FEDesc('id_23', MapEnc({'TRANSPARENT': 4, 'IP_PROXY': 3,
                            'IP_PROXY:ANONYMOUS': 2, 'IP_PROXY:HIDDEN': 1}), astype='category'),
    FEDesc('id_24', NOOP, astype='category'),
    FEDesc('id_25', NOOP, astype='category'),
    FEDesc('id_26', NOOP, astype='category'),
    FEDesc('id_27', MapEnc({'Found': 1, 'NotFound': 0}), astype='category'),
    FEDesc('id_28', MapEnc({'New': 2, 'Found': 1}), astype='category'),
    FEDesc('id_29', MapEnc({'Found': 1, 'NotFound': 0}), astype='category'),
    # FEDesc('id_30', FreqEnc(count_on=C_TR_TE('id_30')), ),  # dst_as='category'),
    # FEDesc('id_31', FreqEnc(count_on=C_TR_TE('id_31')), ),  # dst_as='category'),
    FEDesc('id_32', NOOP, astype='category'),
    # FEDesc('id_33', FreqEnc(count_on=C_TR_TE('id_33')), astype='category'),
    FEDesc('id_34', RegexEnc(r'.*:([-+]?\d+)', do_eval=True), ),  # dst_as='category'),
    FEDesc('id_35', MapEnc({'T': 1, 'F': 0}), astype='category'),
    FEDesc('id_36', MapEnc({'T': 1, 'F': 0}), astype='category'),
    FEDesc('id_37', MapEnc({'T': 1, 'F': 0}), astype='category'),
    FEDesc('id_38', MapEnc({'T': 1, 'F': 0}), astype='category'),

    FEDesc('DeviceType', MapEnc({'desktop': 1, 'mobile': 0}), astype='category'),
    # FEDesc('DeviceInfo', FreqEnc(count_on=C_TR_TE('DeviceInfo'), ), ),  # dst_as='category'),

    # user-defined FE: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    FEDesc('id_02__id_20',              FreqEnc(count_on=C_TR_TE('id_02__id_20')),              dst='id_02__id_20_count'),
    FEDesc('id_02__D8',                 FreqEnc(count_on=C_TR_TE('id_02__D8')),                 dst='id_02__D8_count'),
    FEDesc('card5__P_emaildomain',      FreqEnc(count_on=C_TR_TE('card5__P_emaildomain')),      dst='card5__P_emaildomain_count'),
    FEDesc('card2__id_20',              FreqEnc(count_on=C_TR_TE('card2__id_20')),              dst='card2__id_20_count'),
    FEDesc('card2__dist1',              FreqEnc(count_on=C_TR_TE('card2__dist1')),              dst='card2__dist1_count'),
    FEDesc('card1__card5',              FreqEnc(count_on=C_TR_TE('card1__card5')),              dst='card1__card5_count'),
    FEDesc('addr1__card1',              FreqEnc(count_on=C_TR_TE('addr1__card1')),              dst='addr1__card1_count'),
    FEDesc('P_emaildomain__C2',         FreqEnc(count_on=C_TR_TE('P_emaildomain__C2')),         dst='P_emaildomain__C2_count'),
    FEDesc('DeviceInfo__P_emaildomain', FreqEnc(count_on=C_TR_TE('DeviceInfo__P_emaildomain')), dst='DeviceInfo__P_emaildomain_count'),
    FEDesc('D11__DeviceInfo',           FreqEnc(count_on=C_TR_TE('D11__DeviceInfo')),           dst='D11__DeviceInfo_count'),

    FEDesc('device_name',    LabelEnc(fit_on=C_TR_TE('device_name')),    astype='category'),
    FEDesc('device_version', LabelEnc(fit_on=C_TR_TE('device_version')), astype='category'),
    FEDesc('OS_id_30',       LabelEnc(fit_on=C_TR_TE('OS_id_30')),       astype='category'),
    FEDesc('version_id_30',  LabelEnc(fit_on=C_TR_TE('version_id_30')),  astype='category'),
    FEDesc('browser_type',   LabelEnc(fit_on=C_TR_TE('browser_type')),   astype='category'),
    FEDesc('browser_ver',    LabelEnc(fit_on=C_TR_TE('browser_ver')),    astype='category'),
    FEDesc('browser_for',    LabelEnc(fit_on=C_TR_TE('browser_for')),    astype='category'),
    FEDesc('screen_width', NOOP, ),
    FEDesc('screen_height', NOOP, ),

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

